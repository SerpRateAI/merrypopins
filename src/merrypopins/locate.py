"""
locate.py
---------

Detects pop-ins (sudden displacement jumps) in nano-indentation curves using multiple methods:

• IsolationForest anomaly detection on stiffness and curvature features
• CNN-based autoencoder reconstruction error
• Finite difference method using Fourier spectral analysis
• Savitzky-Golay derivative method

To ensure relevance, all detection methods operate only on the indentation curve **up to the maximum load point**.
This is because pop-in events occur during the loading phase of indentation. After reaching peak load, material unloading
or post-penetration artifacts may dominate, which are irrelevant for pop-in analysis.

None of the methods use pre-trained models. The IsolationForest and the convolutional
autoencoder are both fitted, unsupervised, on the curve being analysed at call time,
so no labelled training data and no shipped weights are required.

The CNN method needs TensorFlow, which is **not** installed with the base package
because it is a large dependency used by only one of the four methods. Install it with::

    pip install 'merrypopins[cnn]'

Provides:
- compute_stiffness
- compute_features
- detect_popins_iforest
- detect_popins_cnn
- detect_popins_fd_fourier
- detect_popins_savgol
- default_locate (combines all methods)
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from tensorflow.keras.models import Model

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

TENSORFLOW_INSTALL_HINT = (
    "The CNN pop-in detector requires TensorFlow, which merrypopins does not "
    "install by default because it is a large dependency used by only one of the "
    "four detection methods. Install it with: pip install 'merrypopins[cnn]'"
)


def _import_keras() -> SimpleNamespace:
    """
    Import the Keras symbols used by the CNN detector.

    TensorFlow is an optional dependency (the ``cnn`` extra), so it is imported
    lazily here rather than at module import time. This keeps the other three
    detection methods, and every other merrypopins module, usable without it.

    Returns:
        SimpleNamespace: Namespace holding the Keras classes used below.

    Raises:
        ImportError: If TensorFlow is not installed, with the install command.
    """
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import Conv1D, Input, MaxPooling1D, UpSampling1D
        from tensorflow.keras.models import Model
    except ImportError as exc:
        raise ImportError(TENSORFLOW_INSTALL_HINT) from exc

    return SimpleNamespace(
        Model=Model,
        Input=Input,
        Conv1D=Conv1D,
        MaxPooling1D=MaxPooling1D,
        UpSampling1D=UpSampling1D,
        EarlyStopping=EarlyStopping,
    )


def compute_stiffness(
    df: pd.DataFrame,
    depth_col: str = "Depth (nm)",
    load_col: str = "Load (µN)",
    window: int = 5,
) -> pd.Series:
    """
    Compute local stiffness (dLoad/dDepth) using sliding-window linear regression.

    In nano-indentation, 'stiffness' is the local slope of the load–depth curve and
    reflects how resistant the material is to deformation. It is computed as:

        stiffness = change in Load / change in Depth
                  = ΔLoad / ΔDepth

    This is estimated using linear regression over a moving window centered on each point.

    Args:
        df (DataFrame): Input indentation data.
        depth_col (str): Column name for depth.
        load_col (str): Column name for load.
        window (int): Sliding window size.

    Returns:
        Series: Stiffness at each data point.
    """
    if depth_col not in df.columns or load_col not in df.columns:
        raise ValueError(
            f"Required columns '{depth_col}' and/or '{load_col}' not found in DataFrame."
        )
    if len(df) < window:
        raise ValueError(
            f"Not enough data points ({len(df)}) for window size ({window})."
        )

    x, y = df[depth_col].values, df[load_col].values
    stiffness = np.full(len(x), np.nan)
    half_win = window // 2

    for i in range(half_win, len(x) - half_win):
        dx = x[i - half_win : i + half_win + 1]
        dy = y[i - half_win : i + half_win + 1]
        A = np.vstack([dx, np.ones_like(dx)]).T
        stiffness[i], _ = np.linalg.lstsq(A, dy, rcond=None)[0]

    return pd.Series(stiffness, index=df.index)


def compute_features(
    df: pd.DataFrame,
    depth_col: str = "Depth (nm)",
    load_col: str = "Load (µN)",
    window: int = 5,
    return_derivatives: bool = True,
) -> pd.DataFrame:
    """
    Compute derived indentation features for anomaly detection.

    This function calculates three features:
      1. Stiffness: local slope of load vs. depth (ΔLoad/ΔDepth)
      2. Stiffness difference: the rate of change in stiffness (first derivative)
      3. Curvature: the rate of change in stiffness difference (second derivative)

    These features help detect sudden shifts in indentation behavior, often indicative
    of pop-in events.

    Args:
        df (DataFrame): Input indentation data.
        depth_col (str): Column name for depth.
        load_col (str): Column name for load.
        window (int): Sliding window size for stiffness calculation.
        return_derivatives (bool): If True (default), return DataFrame with added features.
                                   If False, return original DataFrame without added columns.

    Returns:
        DataFrame: Enhanced DataFrame with 'stiffness', 'stiff_diff', and 'curvature' columns.
    """
    df2 = df.copy()
    df2["stiffness"] = compute_stiffness(df, depth_col, load_col, window)
    df2["stiff_diff"] = df2["stiffness"].diff()
    df2["curvature"] = df2["stiff_diff"].diff()
    return df2 if return_derivatives else df


def find_max_load_index(df: pd.DataFrame, load_col: str = "Load (µN)") -> int:
    """
    Find the index of the maximum load point in the indentation curve.

    Args:
        df (DataFrame): Input indentation data.
        load_col (str): Column name for the load data.

    Returns:
        int: Index of the maximum load value.
    """
    return df[load_col].idxmax()


def trim_edges(series: pd.Series | np.ndarray, margin: int) -> pd.Series | np.ndarray:
    """
    Trim the first `margin` elements of a pandas Series.
    This is useful for removing edge effects in time-series data where
    the first few points may not be reliable.
    Args:
        series (pd.Series): Input time-series data.
        margin (int): Number of elements to trim from the start.
    Returns:
        pd.Series: A copy of the input series with the first `margin` elements set to False.
    """
    trimmed = series.copy()
    trimmed[:margin] = False
    return trimmed


def _apply_trims(
    flags: np.ndarray,
    df: pd.DataFrame,
    load_col: str,
    trim_edges_enabled: bool,
    trim_margin: int | None,
    default_margin: int,
    max_load_trim_enabled: bool,
) -> np.ndarray:
    """
    Apply the two masks every detection method shares.

    Each detector produces raw candidate flags and then discards those that fall in
    the unreliable leading edge of the curve or after the maximum load point. That
    tail is identical across methods and lives here so the four detectors differ only
    in how they find candidates.

    Args:
        flags (ndarray): Boolean candidate flags, one per data point.
        df (DataFrame): The indentation data the flags were computed from.
        load_col (str): Column name for load data.
        trim_edges_enabled (bool): If True, blank out the leading `trim_margin` points.
        trim_margin (int or None): Number of leading elements to blank. If None,
            `default_margin` is used instead.
        default_margin (int): Margin used when `trim_margin` is None. Methods choose
            this to match their own window size.
        max_load_trim_enabled (bool): If True, blank out everything after max load.

    Returns:
        ndarray: The masked flags.
    """
    if trim_edges_enabled:
        margin = trim_margin if trim_margin is not None else default_margin
        flags = trim_edges(flags, margin=margin)

    if max_load_trim_enabled:
        # Mask out anything *after* max load
        max_idx = find_max_load_index(df, load_col)
        flags[max_idx + 1 :] = False

    return flags


def detect_popins_iforest(
    df: pd.DataFrame,
    contamination: float = 0.001,
    random_state: int | None = None,
    depth_col: str = "Depth (nm)",
    load_col: str = "Load (µN)",
    window: int = 5,
    trim_edges_enabled: bool = True,
    trim_margin: int | None = None,
    max_load_trim_enabled: bool = True,
) -> pd.DataFrame:
    """
    Detect pop-ins using Isolation Forest based on local stiffness and curvature.

    This method computes two time-series features:
      - Stiffness difference: the rate of change in the slope of the load–depth curve
      - Curvature: the second derivative of the load curve (change in stiffness difference)

    It then applies the Isolation Forest algorithm from scikit-learn, which isolates
    anomalies by recursively partitioning the feature space. Points that require fewer
    partitions to isolate are more likely to be outliers.

    The forest is fitted on the curve passed in; there is no pre-trained model and no
    labelled training data is required.

    Args:
        df (DataFrame): Indentation dataset containing load and depth columns.
        contamination (float): Proportion of expected anomalies in the dataset.
        random_state (int or None): Random seed for reproducibility.
        depth_col (str): Name of the depth column.
        load_col (str): Name of the load column.
        window (int): Size of the sliding window used to compute stiffness.
        trim_edges_enabled (bool): If True, trims the first `window` elements
        trim_margin (int or None): Number of elements to trim from the start.
        max_load_trim_enabled (bool): If True, masks out any anomalies after the maximum load point. Default is True.

    Returns:
        DataFrame: A copy of the original DataFrame with a new boolean column:
            - "popin_iforest": True for detected pop-ins (anomalies), False otherwise.
                - Only pre-max-load anomalies are returned to focus on loading-phase events. If `max_load_trim_enabled` is True which is the default.
    """
    df2 = compute_features(df, depth_col, load_col, window)
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    features = df2[["stiff_diff", "curvature"]].fillna(0)
    preds = iso.fit_predict(features) == -1

    preds = _apply_trims(
        preds,
        df,
        load_col,
        trim_edges_enabled,
        trim_margin,
        max(10, window),
        max_load_trim_enabled,
    )

    df2["popin_iforest"] = preds
    logger.info(f"IsolationForest flagged {df2['popin_iforest'].sum()} anomalies")
    return df2


def build_cnn_autoencoder(window_size: int, n_features: int) -> Model:
    """
    Build a 1D Convolutional Autoencoder for time-series anomaly detection.

    This model learns to reconstruct input sequences composed of features like
    stiffness difference and curvature. During inference, reconstruction error
    is used to detect anomalies—samples with high error are likely pop-ins.

    Architecture overview:
      - Encoder:
          Conv1D -> MaxPooling -> Conv1D -> MaxPooling -> Conv1D
      - Decoder:
          UpSampling -> Conv1D -> UpSampling -> Conv1D

    The model operates on fixed-size input windows and uses symmetric encoding
    and decoding layers. The final layer has linear activation to match the
    original feature values.

    Args:
        window_size (int): Number of time steps per sequence.
        n_features (int): Number of input features per time step.

    Returns:
        keras.Model: Keras autoencoder model (uncompiled).

    Raises:
        ImportError: If TensorFlow is not installed. Install it with
            `pip install 'merrypopins[cnn]'`.
    """
    keras = _import_keras()

    inp = keras.Input(shape=(window_size, n_features))
    x = keras.Conv1D(32, 3, activation="relu", padding="same")(inp)
    x = keras.MaxPooling1D(2, padding="same")(x)
    x = keras.Conv1D(16, 3, activation="relu", padding="same")(x)
    x = keras.MaxPooling1D(2, padding="same")(x)
    x = keras.Conv1D(16, 3, activation="relu", padding="same")(x)
    x = keras.UpSampling1D(2)(x)
    x = keras.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = keras.UpSampling1D(2)(x)
    x = keras.Conv1D(n_features, 3, activation="linear", padding="same")(x)
    return keras.Model(inp, x)


def detect_popins_cnn(
    df: pd.DataFrame,
    window_size: int = 64,
    epochs: int = 10,
    threshold_multiplier: float = 5.0,
    batch_size: int = 32,
    validation_split: float = 0.0,
    depth_col: str = "Depth (nm)",
    load_col: str = "Load (µN)",
    window: int = 5,
    trim_edges_enabled: bool = True,
    trim_margin: int | None = None,
    max_load_trim_enabled: bool = True,
) -> pd.DataFrame:
    """
    Detect pop-ins using a Convolutional Autoencoder trained on stiffness features.

    This method uses an unsupervised CNN-based autoencoder to learn a compressed
    representation of local indentation behavior. It reconstructs short time windows
    of two features:
      - Stiffness difference: rate of change of the slope (d²Load/dDepth²)
      - Curvature: second derivative of load (d³Load/dDepth³)

    The reconstruction error (mean squared error) is computed between input and output.
    High reconstruction errors indicate patterns that the model considers unusual—
    these are flagged as potential pop-in events.

    The method uses a sliding window to extract overlapping sequences from the full curve,
    trains the model on all windows, and flags windows whose error exceeds a dynamic threshold.
    The autoencoder is trained from scratch on the curve passed in, so no pre-trained
    weights ship with the package and no labelled data is needed.

    Args:
        df (DataFrame): Input indentation data containing load and depth columns.
        window_size (int): Number of time steps per CNN input window.
        epochs (int): Number of training epochs for the autoencoder.
        threshold_multiplier (float): Multiplier for anomaly detection threshold based on std dev.
        batch_size (int): Mini-batch size during training.
        validation_split (float): Proportion of data used for validation (0.0 disables validation).
        depth_col (str): Column name for depth measurements.
        load_col (str): Column name for load measurements.
        window (int): Size of the moving window used for stiffness calculation.
        trim_edges_enabled (bool): If True, trims the first `window` elements
        trim_margin (int or None): Number of elements to trim from the start.
        max_load_trim_enabled (bool): If True, masks out any anomalies after the maximum load point. Default is True.

    Returns:
        DataFrame: Original DataFrame with a new boolean column:
            - "popin_cnn": True for detected anomalies, False otherwise.
                - Only pre-max-load anomalies are returned to focus on loading-phase events. If `max_load_trim_enabled` is True which is the default.

    Raises:
        ImportError: If TensorFlow is not installed. Install it with
            `pip install 'merrypopins[cnn]'`.
    """
    keras = _import_keras()

    df2 = compute_features(df, depth_col, load_col, window)
    X = df2[["stiff_diff", "curvature"]].fillna(0).values
    W = np.array([X[i : i + window_size] for i in range(len(X) - window_size)])

    ae = build_cnn_autoencoder(window_size, 2)
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(
        W,
        W,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=0,
        callbacks=(
            [keras.EarlyStopping(patience=3, restore_best_weights=True)]
            if validation_split > 0
            else None
        ),
    )

    W_pred = ae.predict(W, verbose=0)
    errors = np.mean((W - W_pred) ** 2, axis=(1, 2))
    threshold = errors.mean() + threshold_multiplier * errors.std()

    flags = np.zeros(len(X), dtype=bool)
    flags[window_size // 2 : -window_size // 2] = errors > threshold

    flags = _apply_trims(
        flags,
        df,
        load_col,
        trim_edges_enabled,
        trim_margin,
        max(10, window),
        max_load_trim_enabled,
    )

    df2["popin_cnn"] = flags
    logger.info(f"CNN flagged {df2['popin_cnn'].sum()} anomalies")
    return df2


def detect_popins_fd_fourier(
    df: pd.DataFrame,
    threshold: float = 3.0,
    spacing: float = 1.0,
    trim_edges_enabled: bool = True,
    trim_margin: int | None = None,
    load_col: str = "Load (µN)",
    max_load_trim_enabled: bool = True,
) -> pd.DataFrame:
    """
    Detect pop-ins by estimating the derivative of Load using a Fourier spectral method.

    This method computes the first derivative in the frequency domain using the Fourier Transform.
    The basic idea is that differentiation in the time domain corresponds to multiplying by a frequency
    component in the Fourier domain:

        dLoad/dDepth ≈ IFFT( i * 2π * frequency * FFT(Load) )

    The inverse FFT (IFFT) is then used to convert the differentiated signal back into the spatial domain.
    IFFT takes frequency-domain data and reconstructs the original time-domain (or spatial) signal.

    Anomalies are flagged where the resulting derivative deviates from the mean by more than a
    given number of standard deviations.

    Args:
        df (DataFrame): Input indentation data.
        threshold (float): Std deviation multiplier to flag anomalies in derivative.
        spacing (float): Spacing between data points (in nm or similar units).
        trim_edges_enabled (bool): If True, trims the first `window` elements
        trim_margin (int or None): Number of elements to trim from the start.
        load_col (str): Column name for load data.
        max_load_trim_enabled (bool): If True, masks out any anomalies after the maximum load point. Default is True.

    Returns:
        DataFrame: Original DataFrame with a new boolean column:
            - "popin_fd": True for detected anomalies, False otherwise.
                - Only pre-max-load anomalies are returned to focus on loading-phase events. If `max_load_trim_enabled` is True which is the default.
    """
    load = df[load_col].values
    fft_load = np.fft.fft(load)
    freqs = np.fft.fftfreq(len(load), d=spacing)
    derivative = np.real(np.fft.ifft(1j * 2 * np.pi * freqs * fft_load))
    anomalies = np.abs(derivative - np.mean(derivative)) > threshold * np.std(
        derivative
    )

    anomalies = _apply_trims(
        anomalies,
        df,
        load_col,
        trim_edges_enabled,
        trim_margin,
        10,
        max_load_trim_enabled,
    )

    df2 = df.copy()
    df2["popin_fd"] = anomalies
    logger.info(f"Fourier spectral method flagged {anomalies.sum()} anomalies")
    return df2


def detect_popins_savgol(
    df: pd.DataFrame,
    window_length: int = 11,
    polyorder: int = 2,
    threshold: float = 3.0,
    deriv: int = 1,
    load_col: str = "Load (µN)",
    trim_edges_enabled: bool = True,
    trim_margin: int | None = None,
    max_load_trim_enabled: bool = True,
) -> pd.DataFrame:
    """
    Detect pop-ins using Savitzky-Golay filtered derivatives.

    This method smooths the load data using a polynomial filter and computes its derivative.
    Anomalies are flagged where the derivative differs significantly from its mean value.

    The steps are:
      1. Apply Savitzky-Golay filter to compute the derivative (e.g., velocity or acceleration)
      2. Flag points where |derivative - mean| > threshold * std deviation

    The Savitzky-Golay filter works by fitting successive subsets of adjacent data points
    with a low-degree polynomial using linear least squares.

    Args:
        window_length (int): Length of the filter window (must be odd).
        polyorder (int): Order of polynomial for smoothing.
        threshold (float): Threshold in standard deviations for detecting anomalies.
        deriv (int): Order of derivative to compute (default is 1 for first derivative).
        load_col (str): Column name for load data.
        trim_edges_enabled (bool): If True, trims the first `window` elements
        trim_margin (int or None): Number of elements to trim from the start.
        max_load_trim_enabled (bool): If True, masks out any anomalies after the maximum load point. Default is True.

    Returns:
        - DataFrame: Original dataframe with a new boolean column:
            - "popin_savgol": True for detected anomalies, False otherwise.
                - Only pre-max-load anomalies are returned to focus on loading-phase events. If `max_load_trim_enabled` is True which is the default.
    """
    derivative = savgol_filter(df[load_col], window_length, polyorder, deriv=deriv)
    anomalies = np.abs(derivative - np.mean(derivative)) > threshold * np.std(
        derivative
    )

    anomalies = _apply_trims(
        anomalies,
        df,
        load_col,
        trim_edges_enabled,
        trim_margin,
        max(10, window_length),
        max_load_trim_enabled,
    )

    df2 = df.copy()
    df2["popin_savgol"] = anomalies
    logger.info(f"Savitzky-Golay flagged {anomalies.sum()} anomalies")
    return df2


def default_locate(
    df: pd.DataFrame,
    iforest_contamination: float = 0.001,
    iforest_random_state: int | None = None,
    cnn_window_size: int = 64,
    cnn_epochs: int = 10,
    cnn_threshold_multiplier: float = 5.0,
    cnn_batch_size: int = 32,
    cnn_validation_split: float = 0.0,
    fd_threshold: float = 3.0,
    fd_spacing: float = 1.0,
    savgol_window_length: int = 11,
    savgol_polyorder: int = 2,
    savgol_threshold: float = 3.0,
    sg_deriv_order: int = 1,
    stiffness_window: int = 5,
    trim_edges_enabled: bool = True,
    trim_margin: int | None = None,
    max_load_trim_enabled: bool = True,
    use_iforest: bool = True,
    use_cnn: bool = True,
    use_fd: bool = True,
    use_savgol: bool = True,
    depth_col: str = "Depth (nm)",
    load_col: str = "Load (µN)",
) -> pd.DataFrame:
    """
    Apply all (default) or selected detection methods to identify pop-ins.

    Each enabled method writes its own boolean column ("popin_iforest", "popin_cnn",
    "popin_fd", "popin_savgol"). Those are then combined into three summary columns:

      - "popin": True where *any* enabled method fired (the inclusive union).
      - "popin_score": how many methods fired at that point, i.e. how much the
        methods agree.
      - "popin_confident": True where at least two methods agree, which is the
        column to use when false positives are more costly than missed events.

    The methods are complementary rather than interchangeable. Savitzky-Golay and the
    Fourier derivative are cheap, need only a couple of parameters, and suit a first
    pass over a large batch. The IsolationForest and CNN autoencoder are unsupervised
    and adapt to the data, which helps on noisy curves or unfamiliar materials where a
    fixed derivative threshold does not transfer, at a higher compute cost.

    Note that `use_cnn=True` (the default) needs TensorFlow, an optional dependency.
    Install it with `pip install 'merrypopins[cnn]'`, or pass `use_cnn=False` to run
    the other three methods without it.

    Args:
        df (DataFrame): Input indentation data.
        iforest_contamination (float): Expected contamination level for IsolationForest.
        iforest_random_state (int or None): Seed for reproducibility.
        cnn_window_size (int): Window size for CNN autoencoder.
        cnn_epochs (int): Training epochs for CNN.
        cnn_threshold_multiplier (float): Threshold multiplier for CNN anomaly detection.
        cnn_batch_size (int): Batch size for CNN autoencoder.
        cnn_validation_split (float): Validation split for CNN autoencoder.
        fd_threshold (float): Standard deviation threshold for finite difference method.
        fd_spacing (float): Spacing between samples for FFT derivative.
        savgol_window_length (int): Window size for Savitzky-Golay filter.
        savgol_polyorder (int): Polynomial order for Savitzky-Golay filter.
        savgol_threshold (float): Std deviation threshold for Savitzky-Golay.
        sg_deriv_order (int): Derivative order for Savitzky-Golay.
        stiffness_window (int): Sliding window size for stiffness computation.
        trim_edges_enabled (bool): If True, trims the first `window` elements
        trim_margin (int or None): Number of elements to trim from the start.
        max_load_trim_enabled (bool): If True, masks out any anomalies after the maximum load point. Default is True.
        use_iforest (bool): Whether to use IsolationForest method.
        use_cnn (bool): Whether to use CNN method.
        use_fd (bool): Whether to use finite difference method.
        use_savgol (bool): Whether to use Savitzky-Golay method.
        depth_col (str): Column name for depth data.
        load_col (str): Column name for load data.

    Returns:
        DataFrame: Data with individual method flags, combined flag, and metadata columns.

    Raises:
        ImportError: If `use_cnn` is True and TensorFlow is not installed. Install it
            with `pip install 'merrypopins[cnn]'` or pass `use_cnn=False`.
    """
    df_combined = df.copy()
    method_flags = []

    if use_iforest:
        df_iforest = detect_popins_iforest(
            df,
            contamination=iforest_contamination,
            random_state=iforest_random_state,
            depth_col=depth_col,
            load_col=load_col,
            window=stiffness_window,
            trim_edges_enabled=trim_edges_enabled,
            trim_margin=trim_margin,
            max_load_trim_enabled=max_load_trim_enabled,
        )
        df_combined["popin_iforest"] = df_iforest["popin_iforest"]
        method_flags.append("popin_iforest")

    if use_cnn:
        df_cnn = detect_popins_cnn(
            df,
            window_size=cnn_window_size,
            epochs=cnn_epochs,
            threshold_multiplier=cnn_threshold_multiplier,
            batch_size=cnn_batch_size,
            validation_split=cnn_validation_split,
            depth_col=depth_col,
            load_col=load_col,
            window=stiffness_window,
            trim_edges_enabled=trim_edges_enabled,
            trim_margin=trim_margin,
            max_load_trim_enabled=max_load_trim_enabled,
        )
        df_combined["popin_cnn"] = df_cnn["popin_cnn"]
        method_flags.append("popin_cnn")

    if use_fd:
        df_fd = detect_popins_fd_fourier(
            df,
            threshold=fd_threshold,
            spacing=fd_spacing,
            trim_edges_enabled=trim_edges_enabled,
            trim_margin=trim_margin,
            load_col=load_col,
            max_load_trim_enabled=max_load_trim_enabled,
        )
        df_combined["popin_fd"] = df_fd["popin_fd"]
        method_flags.append("popin_fd")

    if use_savgol:
        df_savgol = detect_popins_savgol(
            df,
            window_length=savgol_window_length,
            polyorder=savgol_polyorder,
            threshold=savgol_threshold,
            deriv=sg_deriv_order,
            load_col=load_col,
            trim_edges_enabled=trim_edges_enabled,
            trim_margin=trim_margin,
            max_load_trim_enabled=max_load_trim_enabled,
        )
        df_combined["popin_savgol"] = df_savgol["popin_savgol"]
        method_flags.append("popin_savgol")

    df_combined["popin"] = df_combined[method_flags].any(axis=1)
    df_combined["popin_methods"] = df_combined[method_flags].apply(
        lambda row: ",".join(
            [col.replace("popin_", "") for col in method_flags if row[col]]
        ),
        axis=1,
    )
    df_combined["popin_score"] = df_combined[method_flags].sum(axis=1)
    df_combined["popin_confident"] = df_combined["popin_score"] >= 2

    total_popins = df_combined["popin"].sum()
    logger.info(f"Total pop-ins detected by selected methods: {total_popins}")

    return df_combined


__all__ = [
    "compute_stiffness",
    "compute_features",
    "detect_popins_iforest",
    "detect_popins_cnn",
    "detect_popins_fd_fourier",
    "detect_popins_savgol",
    "default_locate",
]

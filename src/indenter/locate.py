"""
locate.py
---------

Detects pop-ins (sudden displacement jumps) in nano-indentation curves using multiple methods:

• IsolationForest anomaly detection on stiffness and curvature features
• CNN-based autoencoder reconstruction error
• Finite difference method using Fourier spectral analysis
• Savitzky-Golay derivative method

Provides:
- compute_stiffness
- compute_features
- detect_popins_iforest
- detect_popins_cnn
- detect_popins_fd_fourier
- detect_popins_savgol
- default_locate (combines all methods)
"""

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest
from scipy.signal import savgol_filter
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def compute_stiffness(df, depth_col="Depth (nm)", load_col="Load (µN)", window=5):
    r"""
    Compute local stiffness (dLoad/dDepth) using sliding-window linear regression:

    \[
    \text{stiffness} = \frac{\Delta \text{Load}}{\Delta \text{Depth}}
    \]

    Args:
        df (DataFrame): Input indentation data.
        depth_col (str): Column name for depth.
        load_col (str): Column name for load.
        window (int): Sliding window size.

    Returns:
        Series: Stiffness at each data point.
    """
    if depth_col not in df.columns or load_col not in df.columns:
        raise ValueError(f"Required columns '{depth_col}' and/or '{load_col}' not found in DataFrame.")
    if len(df) < window:
        raise ValueError(f"Not enough data points ({len(df)}) for window size ({window}).")

    x, y = df[depth_col].values, df[load_col].values
    stiffness = np.full(len(x), np.nan)
    half_win = window // 2

    for i in range(half_win, len(x) - half_win):
        dx = x[i - half_win:i + half_win + 1]
        dy = y[i - half_win:i + half_win + 1]
        A = np.vstack([dx, np.ones_like(dx)]).T
        stiffness[i], _ = np.linalg.lstsq(A, dy, rcond=None)[0]

    return pd.Series(stiffness, index=df.index)

def compute_features(df, depth_col="Depth (nm)", load_col="Load (µN)", window=5, return_derivatives=True):
    """
    Compute stiffness, stiffness difference, and curvature features.

    Args:
        df (DataFrame): Input indentation data.
        depth_col (str): Column name for depth.
        load_col (str): Column name for load.
        window (int): Sliding window size for stiffness calculation.
        return_derivatives (bool): If True (default), return derivatives; otherwise, return original DataFrame.

    Returns:
        DataFrame: Enhanced DataFrame with stiffness, stiff_diff, and curvature.
    """
    df2 = df.copy()
    df2["stiffness"] = compute_stiffness(df, depth_col, load_col, window)
    df2["stiff_diff"] = df2["stiffness"].diff()
    df2["curvature"] = df2["stiff_diff"].diff()
    return df2 if return_derivatives else df

def detect_popins_iforest(df, contamination=0.001, random_state=None, depth_col="Depth (nm)", load_col="Load (µN)", window=5):
    """
    Detect pop-ins using Isolation Forest based on stiffness difference and curvature.

    This method applies scikit-learn's IsolationForest to identify anomalies
    in a 2D feature space composed of local stiffness changes and curvature.

    Args:
        df (DataFrame): Indentation data.
        contamination (float): Expected fraction of anomalies.
        random_state (int or None): Seed for reproducibility.
        depth_col (str): Column name for depth.
        load_col (str): Column name for load.
        window (int): Sliding window size for stiffness calculation.

    Returns:
        DataFrame: Original DataFrame with an added 'popin_iforest' boolean column
                   marking detected anomalies.
    """
    df2 = compute_features(df, depth_col=depth_col, load_col=load_col, window=window)
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    features = df2[["stiff_diff", "curvature"]].fillna(0)
    df2["popin_iforest"] = iso.fit_predict(features) == -1
    logger.info(f"IsolationForest flagged {df2['popin_iforest'].sum()} anomalies")
    return df2

def build_cnn_autoencoder(window_size, n_features):
    """
    Build a 1D Convolutional Autoencoder model for anomaly detection.

    The model compresses and reconstructs time-series features using the architecture:
    Conv1D → MaxPooling → Conv1D → MaxPooling → Conv1D → UpSampling → Conv1D → UpSampling → Conv1D

    Args:
        window_size (int): Length of the input time window.
        n_features (int): Number of input features (e.g., 2 for [stiff_diff, curvature]).

    Returns:
        Model: Compiled Keras model for training and inference.
    """
    inp = Input(shape=(window_size, n_features))
    x = Conv1D(32, 3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(32, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(n_features, 3, activation='linear', padding='same')(x)
    return Model(inp, x)

def detect_popins_cnn(df, window_size=64, epochs=10, threshold_multiplier=5.0, batch_size=32, validation_split=0.0, depth_col="Depth (nm)", load_col="Load (µN)", window=5):
    """
    Detect pop-ins using a CNN autoencoder based on local stiffness and curvature.
    This method trains a convolutional autoencoder to reconstruct sliding windows of
    stiffness difference and curvature features, then flags anomalies based on reconstruction error.
    
    Args:
        df (DataFrame): Input indentation data.
        window_size (int): Size of the sliding window for CNN input.
        epochs (int): Number of training epochs.
        threshold_multiplier (float): Multiplier for setting anomaly detection threshold.
        batch_size (int): Batch size for training.
        validation_split (float): Fraction of data to use for validation.
        depth_col (str): Column name for depth.
        load_col (str): Column name for load.
        window (int): Sliding window size for stiffness calculation.
    
    Returns:
        DataFrame: Original DataFrame with an added 'popin_cnn' boolean column marking detected anomalies.
    """
    df2 = compute_features(df, depth_col=depth_col, load_col=load_col, window=window)
    X = df2[["stiff_diff", "curvature"]].fillna(0).values
    W = np.array([X[i:i+window_size] for i in range(len(X)-window_size)])

    ae = build_cnn_autoencoder(window_size, 2)
    ae.compile(optimizer='adam', loss='mse')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ae.fit(W, W, epochs=epochs, batch_size=batch_size, validation_split=validation_split,
           callbacks=[early_stopping] if validation_split > 0 else None, verbose=0)

    W_pred = ae.predict(W, verbose=0)
    errors = np.mean((W - W_pred)**2, axis=(1,2))
    threshold = errors.mean() + threshold_multiplier * errors.std()

    flags = np.zeros(len(X), dtype=bool)
    flags[window_size//2:-window_size//2] = errors > threshold
    df2["popin_cnn"] = flags
    logger.info(f"CNN flagged {flags.sum()} anomalies")
    return df2

def detect_popins_fd_fourier(df, threshold=3.0, spacing=1.0):
    r"""
    Detect pop-ins by estimating the derivative of Load using a Fourier spectral method.

    This method computes the first derivative in the frequency domain via FFT:
        \[ \frac{df}{dx} \approx \mathcal{F}^{-1}\left( i 2 \pi f \cdot \mathcal{F}(f) \right) \]

    Args:
        df (DataFrame): Input indentation data.
        threshold (float): Std deviation multiplier to flag anomalies in derivative.
        spacing (float): Spacing between data points (in nm or similar units).

    Returns:
        DataFrame: Copy of input with "popin_fd" flag column.
    """
    load = df["Load (µN)"].values
    n = len(load)
    fft_load = np.fft.fft(load)
    freqs = np.fft.fftfreq(n, d=spacing)
    fft_derivative = 1j * 2 * np.pi * freqs * fft_load
    derivative = np.real(np.fft.ifft(fft_derivative))
    mean, std = np.mean(derivative), np.std(derivative)
    anomalies = np.abs(derivative - mean) > threshold * std
    df2 = df.copy()
    df2["popin_fd"] = anomalies
    logger.info(f"Fourier spectral method flagged {anomalies.sum()} anomalies")
    return df2

def detect_popins_savgol(df, window_length=11, polyorder=2, threshold=3.0, deriv=1, load_col="Load (µN)"):
    """
    Detect pop-ins using Savitzky-Golay filtered derivatives.

    Args:
        window_length (int): Length of the filter window.
        polyorder (int): Order of polynomial for smoothing.
        threshold (float): Threshold in standard deviations for detecting anomalies.
        deriv (int): Order of derivative to compute (default is 1 for first derivative).
        load_col (str): Column name for load data.

    Returns:
        DataFrame: Original data with popin_savgol column.
    """
    derivative = savgol_filter(df[load_col], window_length, polyorder, deriv=deriv)
    mean, std = np.mean(derivative), np.std(derivative)
    anomalies = np.abs(derivative - mean) > threshold * std
    df2 = df.copy()
    df2["popin_savgol"] = anomalies
    logger.info(f"Savitzky-Golay flagged {anomalies.sum()} anomalies")
    return df2

# default_locate implementation combining all methods with popin flag.
...

def default_locate(
    df,
    iforest_contamination=0.001,
    iforest_random_state=None,
    cnn_window_size=64,
    cnn_epochs=10,
    cnn_threshold_multiplier=5.0,
    cnn_batch_size=32,
    cnn_validation_split=0.0,
    fd_threshold=3.0,
    fd_spacing=1.0,
    savgol_window_length=11,
    savgol_polyorder=2,
    savgol_threshold=3.0,
    sg_deriv_order=1,
    stiffness_window=5,
    use_iforest=True,
    use_cnn=True,
    use_fd=True,
    use_savgol=True,
    depth_col="Depth (nm)",
    load_col="Load (µN)"
):
    """
    Apply all (default) or selected detection methods to identify pop-ins.

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
        use_iforest (bool): Whether to use IsolationForest method.
        use_cnn (bool): Whether to use CNN method.
        use_fd (bool): Whether to use finite difference method.
        use_savgol (bool): Whether to use Savitzky-Golay method.
        depth_col (str): Column name for depth data.
        load_col (str): Column name for load data.

    Returns:
        DataFrame: Data with individual method flags, combined flag, and metadata columns.
    """
    df_combined = df.copy()
    method_flags = []

    if use_iforest:
        df_iforest = detect_popins_iforest(df, contamination=iforest_contamination, random_state=iforest_random_state, depth_col=depth_col, load_col=load_col, window=stiffness_window)
        df_combined["popin_iforest"] = df_iforest["popin_iforest"]
        method_flags.append("popin_iforest")

    if use_cnn:
        df_cnn = detect_popins_cnn(df, window_size=cnn_window_size, epochs=cnn_epochs, threshold_multiplier=cnn_threshold_multiplier, batch_size=cnn_batch_size, validation_split=cnn_validation_split, depth_col=depth_col, load_col=load_col, window=stiffness_window)
        df_combined["popin_cnn"] = df_cnn["popin_cnn"]
        method_flags.append("popin_cnn")

    if use_fd:
        df_fd = detect_popins_fd_fourier(df, threshold=fd_threshold, spacing=fd_spacing)
        df_combined["popin_fd"] = df_fd["popin_fd"]
        method_flags.append("popin_fd")

    if use_savgol:
        df_savgol = detect_popins_savgol(df, window_length=savgol_window_length, polyorder=savgol_polyorder, threshold=savgol_threshold, deriv=sg_deriv_order, load_col=load_col)
        df_combined["popin_savgol"] = df_savgol["popin_savgol"]
        method_flags.append("popin_savgol")

    df_combined['popin'] = df_combined[method_flags].any(axis=1)
    df_combined['popin_methods'] = df_combined[method_flags].apply(lambda row: ','.join([col.replace('popin_', '') for col in method_flags if row[col]]), axis=1)
    df_combined['popin_score'] = df_combined[method_flags].sum(axis=1)
    df_combined['popin_confident'] = df_combined['popin_score'] >= 2

    total_popins = df_combined['popin'].sum()
    logger.info(f"Total pop-ins detected by selected methods: {total_popins}")

    return df_combined

__all__ = [
    "compute_stiffness", 
    "compute_features", 
    "detect_popins_iforest", 
    "detect_popins_cnn",
    "detect_popins_fd_fourier", 
    "detect_popins_savgol",
    "default_locate"
]

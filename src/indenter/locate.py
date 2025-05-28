# src/indenter/locate.py

"""
locate.py
---------

Detects pop-ins (sudden displacement jumps) in nano-indentation curves using two unsupervised methods:
  - IsolationForest anomaly detection on slope/curvature features
  - 1D convolutional autoencoder reconstruction error

Functions:
  - compute_stiffness: estimate local stiffness via sliding-window regression
  - compute_features: derive first and second derivatives (stiff_diff, curvature)
  - detect_popins_iforest: flag anomalies from IsolationForest
  - build_cnn_autoencoder: construct the Conv1D autoencoder model
  - detect_popins_cnn: flag anomalies via autoencoder reconstruction error
  - default_locate: run both detectors, combine masks, and report overlap
"""
# Import necessary libraries
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D

# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)


def compute_stiffness(
    df: pd.DataFrame,
    depth_col: str = "Depth (nm)",
    load_col: str = "Load (µN)",
    window: int = 5
) -> pd.Series:
    """
    Estimate local stiffness dLoad/dDepth via sliding-window regression.

    Returns a Series of same length (edges filled with NaN).
    """
    depth = df[depth_col].values.reshape(-1, 1)
    load = df[load_col].values
    n = len(depth)
    k = window // 2
    stiff = np.full(n, np.nan)

    # slide a local linear fit
    for i in range(k, n - k):
        x = depth[i - k : i + k + 1]
        y = load[i - k : i + k + 1]
        A = np.hstack([x, np.ones_like(x)])
        m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        stiff[i] = m

    return pd.Series(stiff, index=df.index)


def compute_features(
    df: pd.DataFrame,
    depth_col: str = "Depth (nm)",
    load_col: str = "Load (µN)",
    window: int = 5
) -> pd.DataFrame:
    """
    Compute slope-based and curvature features required for anomaly detectors:
      - 'stiffness': local dLoad/dDepth
      - 'stiff_diff': first difference of stiffness
      - 'curvature': second difference (difference of stiffness differences)
    """
    df2 = df.copy()
    df2['stiffness'] = compute_stiffness(df, depth_col, load_col, window)
    df2['stiff_diff'] = df2['stiffness'].diff()
    df2['curvature'] = df2['stiff_diff'].diff()
    return df2


def detect_popins_iforest(
    df: pd.DataFrame,
    features=('stiff_diff', 'curvature'),
    contamination: float = 0.005,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Unsupervised pop-in detection via IsolationForest.
    Flags ~contamination fraction of points as anomalies based on provided features.
    Adds column 'popin_iforest' to output.
    """
    df2 = compute_features(df)
    X = df2[list(features)].fillna(0).values
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state
    )
    labels = iso.fit_predict(X)
    # -1 => anomaly
    df2['popin_iforest'] = labels == -1
    n = df2['popin_iforest'].sum()
    logger.info(f"detect_popins_iforest: flagged {n} anomalies via IsolationForest")
    return df2


def build_cnn_autoencoder(
    window_size: int,
    n_features: int,
    latent_dim: int = 16
) -> Model:
    """
    Build a simple 1D convolutional autoencoder for windowed data.
    A 1D model is chosen because indentation data are sequential time-series along depth.
    """
    inp = Input(shape=(window_size, n_features))
    # Encoder
    x = Conv1D(32, 3, activation='relu', padding='same')(inp)
    x = MaxPooling1D(2, padding='same')(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = MaxPooling1D(2, padding='same')(x)
    # Bottleneck
    x = Conv1D(latent_dim, 3, activation='relu', padding='same')(x)
    # Decoder
    x = UpSampling1D(2)(x)
    x = Conv1D(16, 3, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(n_features, 3, activation='linear', padding='same')(x)

    return Model(inputs=inp, outputs=x)


def detect_popins_cnn(
    df: pd.DataFrame,
    features=('stiff_diff', 'curvature'),
    window_size: int = 64,
    latent_dim: int = 16,
    epochs: int = 10,
    batch_size: int = 32,
    error_multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Unsupervised pop-in detection via 1D CNN autoencoder on sliding windows.
    - Build windows of length `window_size` from the feature time series.
    - Train the autoencoder to reconstruct each window.
    - Compute reconstruction MSE per window and assign it to the center index.
    - Flag points with error > mean + error_multiplier*std as anomalies.
    Adds 'popin_cnn' and 'ae_error' columns to output.
    """
    df2 = compute_features(df)
    X_feat = df2[list(features)].fillna(0).values
    n_samples, n_feat = X_feat.shape
    # create sliding windows
    windows = []
    for i in range(n_samples - window_size + 1):
        windows.append(X_feat[i : i + window_size])
    Xw = np.stack(windows, axis=0)

    # build & train AE
    ae = build_cnn_autoencoder(window_size, n_feat, latent_dim)
    ae.compile(optimizer='adam', loss='mse')
    ae.fit(Xw, Xw, epochs=epochs, batch_size=batch_size, verbose=0)

    # reconstruct & compute error
    Xw_pred = ae.predict(Xw, batch_size=batch_size, verbose=0)
    errors = np.mean((Xw - Xw_pred) ** 2, axis=(1, 2))

    # map window errors to center indices
    errs_full = np.zeros(n_samples)
    counts = np.zeros(n_samples)
    for idx, err in enumerate(errors):
        center = idx + window_size // 2
        errs_full[center] += err
        counts[center] += 1
    # average if multiple
    mask = counts > 0
    errs_full[mask] /= counts[mask]

    df2['ae_error'] = errs_full
    μ, σ = errs_full[mask].mean(), errs_full[mask].std()
    threshold = μ + error_multiplier * σ
    df2['popin_cnn'] = df2['ae_error'] > threshold

    n = df2['popin_cnn'].sum()
    logger.info(f"detect_popins_cnn: flagged {n} anomalies via CNN autoencoder")
    return df2


def default_locate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run both IsolationForest and CNN-AE detectors and combine results.
    Adds 'popin_iforest', 'popin_cnn', 'popin_both', and final 'popin' mask.
    Also logs the percentage of pop-ins identified by both methods.
    """
    df_if = detect_popins_iforest(df)
    df_cnn = detect_popins_cnn(df)
    # ensure same index and merge flags
    df2 = df.copy()
    df2['popin_iforest'] = df_if['popin_iforest']
    df2['popin_cnn']      = df_cnn['popin_cnn']
    # both detectors agree
    df2['popin_both']     = df2['popin_iforest'] & df2['popin_cnn']
    # union of any detector
    df2['popin']          = df2['popin_iforest'] | df2['popin_cnn']
    total = df2['popin'].sum()
    both = df2['popin_both'].sum()
    pct = (both / total * 100) if total > 0 else 0
    logger.info(f"default_locate: total pop-ins flagged = {total}; overlap = {both} ({pct:.1f}% overlap)")
    return df2

__all__ = [
    'build_cnn_autoencoder',
    'compute_stiffness',
    'compute_features',
    'detect_popins_iforest',
    'detect_popins_cnn',
    'default_locate'
]

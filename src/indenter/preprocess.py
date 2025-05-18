"""
preprocess.py
------------

Provides pre-processing functions for indentation datasets.

Functions:
    - remove_initial_data: remove all points up to minimum Load point.
    - rescale_data: automatically detect contact point and rescale Depth.
    - preprocess_pipeline: apply both steps in sequence.

Usage:
    from indenter.preprocess import remove_initial_data, rescale_data, preprocess_pipeline
"""

import pandas as pd
import numpy as np
import logging
from scipy.signal import savgol_filter

# Module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def remove_pre_min_load(df: pd.DataFrame, load_col="Load (µN)") -> pd.DataFrame:
    """
    Remove all points up to and including the minimum Load point.

    Args:
        df (pd.DataFrame): Input DataFrame.
        load_col (str): Load column name.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df2 = df.copy()
    loads = df2[load_col].values
    min_idx = np.argmin(loads)

    if min_idx >= len(loads) - 1:
        logger.warning("Minimum at end of data; skipping initial data removal.")
        return df2

    df_clean = df2.iloc[min_idx + 1:].reset_index(drop=True)
    logger.info(f"Removed first {min_idx + 1} points up to minimum Load ({loads[min_idx]:.2f})")
    return df_clean

def rescale_data(
    df: pd.DataFrame,
    depth_col="Depth (nm)",
    load_col="Load (µN)",
    N_baseline=50,
    k=5,
    window_length=11,
    polyorder=2
) -> pd.DataFrame:
    """
    Automatically detect contact point by noise threshold and rescale Depth to zero.

    Args:
        df (pd.DataFrame): Input DataFrame.
        depth_col (str): Depth column name.
        load_col (str): Load column name.
        N_baseline (int): Number of points for baseline noise estimation.
        k (float): Noise multiplier for threshold.
        window_length (int): Smoothing window (must be odd).
        polyorder (int): Polynomial order for smoothing.

    Returns:
        pd.DataFrame: Rescaled DataFrame.
    """
    df2 = df.copy()
    loads = df2[load_col].values
    baseline = loads[:N_baseline]
    noise_mean, noise_std = baseline.mean(), baseline.std()
    threshold = noise_mean + k * noise_std

    wl = min(window_length, len(loads) // 2 * 2 - 1)
    smooth_loads = savgol_filter(loads, window_length=wl, polyorder=polyorder)

    idx = np.argmax(smooth_loads > threshold)
    if smooth_loads[idx] <= threshold:
        logger.warning(f"No crossing above auto-threshold ({threshold:.2f}); skipping rescale.")
        return df2

    shift = df2[depth_col].iloc[idx]
    df2[depth_col] = df2[depth_col] - shift
    logger.info(f"Auto-rescaled at index {idx}, load={smooth_loads[idx]:.2f} > {threshold:.2f}, shift={shift:.1f} nm")
    return df2

def preprocess_pipeline(
    df: pd.DataFrame,
    depth_col="Depth (nm)",
    load_col="Load (µN)",
    N_baseline=50,
    k=5,
    window_length=11,
    polyorder=2
) -> pd.DataFrame:
    """
    Full pre-processing pipeline for indentation data:
    1. Remove all points up to and including the minimum Load point.
    2. Rescale Depth so that the detected contact point is zero.

    Args:
        df (pd.DataFrame): Input DataFrame.
        depth_col (str): Depth column name.
        load_col (str): Load column name.
        N_baseline (int): Baseline points for rescaling.
        k (float): Threshold multiplier.
        window_length (int): Smoothing window.
        polyorder (int): Smoothing polynomial order.

    Returns:
        pd.DataFrame: pre-processed DataFrame.
    """
    df_cleaned = remove_initial_data(df, load_col=load_col)
    df_rescaled = rescale_data(
        df_cleaned,
        depth_col=depth_col,
        load_col=load_col,
        N_baseline=N_baseline,
        k=k,
        window_length=window_length,
        polyorder=polyorder
    )
    return df_rescaled

# package exports
__all__ = [
    'remove_initial_data',
    'rescale_data',
    'preprocess_pipeline'
]

"""
preprocess.py
-------------

Provides pre-processing functions for indentation datasets.

Functions:
    - remove_pre_min_load: Remove all points up to the minimum Load point.
    - rescale_data: Automatically detect contact point and rescale Depth.
    - finalise_contact_index: Optionally trim and/or flag the contact point.
    - default_preprocess: Recommended preprocessing pipeline.

Usage:
    from merrypopins.preprocess import (
        remove_pre_min_load,
        rescale_data,
        finalise_contact_index,
        default_preprocess
    )
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

    df_clean = df2.iloc[min_idx + 1 :].reset_index(drop=True)
    logger.info(
        f"Removed first {min_idx + 1} points up to minimum Load ({loads[min_idx]:.2f})"
    )
    return df_clean


def rescale_data(
    df: pd.DataFrame,
    depth_col="Depth (nm)",
    load_col="Load (µN)",
    N_baseline=50,
    k=5,
    window_length=11,
    polyorder=2,
) -> pd.DataFrame:
    """
    Automatically detect contact point by noise threshold and rescale Depth so contact = 0.

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

    # Safely calculate smoothing window
    wl = min(window_length, len(loads) // 2 * 2 - 1)

    if wl <= polyorder:
        logger.warning("Not enough data to smooth: skipping smoothing.")
        smooth_loads = loads
    else:
        smooth_loads = savgol_filter(loads, window_length=wl, polyorder=polyorder)

    idx = np.argmax(smooth_loads > threshold)
    if smooth_loads[idx] <= threshold:
        logger.warning(
            f"No crossing above auto-threshold ({threshold:.2f}); skipping rescale."
        )
        return df2

    shift = df2[depth_col].iloc[idx]
    df2[depth_col] = df2[depth_col] - shift
    logger.info(
        f"Auto-rescaled at index {idx}, load={smooth_loads[idx]:.2f} > {threshold:.2f}, shift={shift:.1f} nm"
    )
    return df2


def finalise_contact_index(
    df: pd.DataFrame,
    depth_col: str = "Depth (nm)",
    remove_pre_contact: bool = True,
    add_flag_column: bool = True,
    flag_column: str = "contact_point",
) -> pd.DataFrame:
    """
    Optionally remove all rows before contact (Depth < 0) and/or flag the first contact point.

    Args:
        df (pd.DataFrame): Rescaled DataFrame.
        depth_col (str): Depth column name.
        remove_pre_contact (bool): If True, remove rows with Depth < 0. Default is True.
        add_flag_column (bool): If True, add a column marking the contact index. Default is True.
        flag_column (str): Name of the column used to flag the contact point. Default column name is "contact_point".

    Returns:
        pd.DataFrame: DataFrame after trimming/flagging contact point.
    """
    df2 = df.copy()
    contact_idx = df2[df2[depth_col] >= 0].index.min()

    if pd.isna(contact_idx):
        if add_flag_column:
            df2[flag_column] = False
        if remove_pre_contact:
            df2 = df2.iloc[0:0]
        logger.warning("No Depth >= 0 found; contact index undefined.")
        return df2

    if add_flag_column:
        df2[flag_column] = False
        df2.loc[contact_idx, flag_column] = True
        logger.info(f"Flagged contact point at index {contact_idx}")

    if remove_pre_contact:
        df2 = df2.loc[contact_idx:].reset_index(drop=True)
        logger.info(f"Removed {contact_idx} rows before contact point (Depth < 0)")

    return df2


def default_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Default preprocessing pipeline using recommended settings.

    Steps:
        - Remove early data up to the minimum Load point
        - Automatically detect contact and rescale Depth
        - Remove Depth < 0 rows and flag the contact point

    Args:
        df (pd.DataFrame): Raw indentation data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = remove_pre_min_load(df, load_col="Load (µN)")
    df = rescale_data(
        df,
        depth_col="Depth (nm)",
        load_col="Load (µN)",
        N_baseline=50,
        k=5,
        window_length=11,
        polyorder=2,
    )
    df = finalise_contact_index(
        df,
        depth_col="Depth (nm)",
        remove_pre_contact=True,
        add_flag_column=True,
        flag_column="contact_point",
    )
    return df


# package exports
__all__ = [
    "remove_pre_min_load",
    "rescale_data",
    "finalise_contact_index",
    "default_preprocess",
]

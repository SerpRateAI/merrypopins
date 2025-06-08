"""
statistics.py
-------------

Extracts statistics from nanoindentation data:
- Postprocess popin candidate flags
- Extract pop-in intervals from candidate flags
- Calculate pop-in statistics
- Stress–strain transformation (Dao et al. 2008)
""" 

import numpy as np
import pandas as pd
from scipy.stats import linregress
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def postprocess_popins_local_max(df, popin_flag_column="popin", window=1):
    """
    Select local load maxima (true pop-ins) before max load.

    Parameters
    ----------
    df : pd.DataFrame
        Input indentation data with pop-in flag.
    popin_flag_column : str
        Column with boolean pop-in detection.
    window : int
        Local window around each point to assess max.

    Returns
    -------
    pd.DataFrame
        Copy with new boolean column: popin_selected
    """
    df = df.copy()
    max_load_idx = df["Load (µN)"].idxmax()
    popin_flags = df[popin_flag_column] == True
    selected_indices = []

    for idx in df.index[window:-window]:
        if idx >= max_load_idx:
            break
        if not popin_flags.loc[idx]:
            continue
        prev_load = df.at[idx - window, "Load (µN)"]
        curr_load = df.at[idx, "Load (µN)"]
        next_load = df.at[idx + window, "Load (µN)"]
        if curr_load > prev_load and curr_load > next_load:
            selected_indices.append(idx)

    df["popin_selected"] = False
    df.loc[selected_indices, "popin_selected"] = True
    logger.info(f"Filtered to {len(selected_indices)} local max pop-ins before max load")
    return df



def extract_popin_intervals(df, popin_col="popin_selected", load_col="Load (µN)"):
    """
    Extract start and end indices for each pop-in event.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with boolean pop-in column.
    popin_col : str
        Column marking pop-in points.
    load_col : str
        Load column used to find recovery point.

    Returns
    -------
    pd.DataFrame
        Copy with start_idx and end_idx columns.
    """
    start_idx_col = [None] * len(df)
    end_idx_col = [None] * len(df)
    popin_indices = df.index[df[popin_col]].tolist()

    for start_idx in popin_indices:
        load_start = df.at[start_idx, load_col]
        end_idx = start_idx
        for i in range(start_idx + 1, len(df)):
            if df.at[i, load_col] >= load_start:
                end_idx = i
                break
        start_idx_col[start_idx] = start_idx
        end_idx_col[start_idx] = end_idx

    df = df.copy()
    df["start_idx"] = start_idx_col
    df["end_idx"] = end_idx_col
    return df



def calculate_popin_statistics(
    df,
    general_stats=True,
    precursor_stats=True,
    temporal_stats=True,
    popin_shape_stats=True,
    time_col="Time (s)",
    load_col="Load (µN)",
    depth_col="Depth (nm)",
    start_col="start_idx",
    end_col="end_idx",
    before_window=0.5,
    after_window=0.5
):
    """
    Compute selected statistics for pop-in intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Data with start/end indices of pop-ins.
    general_stats : bool
        Compute overall curve-level stats (e.g. n_popins, total duration).
    precursor_stats : bool
        Compute indicators before pop-in onset.
    temporal_stats : bool
        Compute timing-based statistics (duration, intervals).
    popin_shape_stats : bool
        Compute features related to the popin shape.
    time_col, load_col, depth_col : str
        Column names for time, load, and depth.
    start_col, end_col : str
        Column names for pop-in start and end indices.
    before_window, after_window : float
        Time windows before and after pop-in.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns.
    """
    df = df.copy()
    interval_rows = df.dropna(subset=[start_col, end_col]).copy().reset_index(drop=True)
    df["dLoad"] = df[load_col].diff() / df[time_col].diff()

    # General stats block (per curve)
    if general_stats:
        n_popins = len(interval_rows)
        df["n_popins"] = n_popins

        if n_popins > 0:
            all_starts = interval_rows[start_col].astype(int).apply(lambda idx: df.at[idx, time_col])
            all_ends = interval_rows[end_col].astype(int).apply(lambda idx: df.at[idx, time_col])
            total_popin_duration = all_ends.max() - all_starts.min()
            avg_time_between = all_starts.diff().dropna().mean()
            first_popin_time = all_starts.min()
            last_popin_time = all_ends.max()
        else:
            total_popin_duration = 0.0
            avg_time_between = None
            first_popin_time = None
            last_popin_time = None

        test_duration = df[time_col].max() - df[time_col].min()
        df["total_test_duration"] = test_duration
        df["total_popin_duration"] = total_popin_duration
        df["first_popin_time"] = first_popin_time
        df["last_popin_time"] = last_popin_time
        df["avg_time_between_popins"] = avg_time_between

    # Per-pop-in stats
    results = []

    for i, row in interval_rows.iterrows():
        start_idx = int(row[start_col])
        end_idx = int(row[end_col])

        start_time = df.at[start_idx, time_col]
        end_time = df.at[end_idx, time_col]

        record = {
            "start_idx": start_idx,
            "end_idx": end_idx
        }

        before = df[(df[time_col] >= start_time - before_window) & (df[time_col] < start_time)]
        during = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]

        def slope_or_none(subset):
            if len(subset) > 1:
                return linregress(subset[time_col], subset[load_col]).slope
            return None

        if temporal_stats:
            popin_length = end_time - start_time
            time_until_next = None
            if i < len(interval_rows) - 1:
                next_start_idx = int(interval_rows.at[i + 1, start_col])
                next_start_time = df.at[next_start_idx, time_col]
                time_until_next = next_start_time - end_time

            record.update({
                "popin_length": popin_length,
                "time_until_next": time_until_next,
                "avg_time_during": during[time_col].mean() if not during.empty else None
            })

        if precursor_stats:
            record.update({
                "avg_dload_before": before["dLoad"].mean() if not before.empty else None,
                "slope_before": slope_or_none(before)
            })

        if popin_shape_stats:
            t_vals = during[time_col].values
            h_vals = during[depth_col].values

            # Compute average depth velocity and curvature
            if len(h_vals) > 2:
                d_depth = np.diff(h_vals)
                d_time = np.diff(t_vals)
                avg_velocity = np.mean(d_depth / d_time)

                curvature = np.gradient(np.gradient(h_vals, t_vals), t_vals)
                avg_curvature = np.mean(curvature)
            else:
                avg_velocity = None
                avg_curvature = None

            record.update({
                "depth_jump": df.at[end_idx, depth_col] - df.at[start_idx, depth_col],
                "avg_depth_during": during[depth_col].mean() if not during.empty else None,
                "avg_depth_velocity": avg_velocity,
                "avg_curvature_depth": avg_curvature
            })

        results.append(record)

    stats_df = pd.DataFrame(results)
    for col in stats_df.columns:
        if col not in [start_col, end_col]:
            df[col] = df[start_col].map(stats_df.set_index("start_idx")[col])

    logger.info(f"Computed pop-in statistics for {len(stats_df)} pop-ins")
    return df


def default_statistics(df_locate, popin_flag_column="popin", before_window=0.5, after_window=0.5):
    """
    Pipeline for pop-in statistics from load–depth domain.

    Parameters
    ----------
    df_locate : pd.DataFrame
        With pop-in candidate flags and load/time/depth.
    popin_flag_column : str
        Column name indicating pop-in detection flag.
    before_window : float
        Time window before pop-in.
    after_window : float
        Time window after pop-in.

    Returns
    -------
    pd.DataFrame
        Copy with popin intervals and basic stats.
    """
    required_cols = ["Time (s)", "Load (µN)", "Depth (nm)", popin_flag_column]
    if "contact_point" in df_locate.columns:
        required_cols.append("contact_point")
    df_locate = df_locate[required_cols].copy()

    df1 = postprocess_popins_local_max(df_locate, popin_flag_column=popin_flag_column)
    df2 = extract_popin_intervals(df1)
    return calculate_popin_statistics(df2, time_col="Time (s)")


__all__ = [
    "postprocess_popins_local_max",
    "extract_popin_intervals",
    "calculate_popin_statistics",
    "calculate_stress_strain",
    "default_statistics"
]

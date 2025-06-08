"""
statistics.py
-------------

Extracts statistics from nanoindentation data:
- Postprocess popin candidate flags
- Extract pop-in intervals from candidate flags
- Calculate pop-in statistics
- Calculate curve-level summary statistics about pop-in activity
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
    method_column : str
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

def _compute_temporal_stats(start_time, end_time, interval_rows, i, df, time_col, start_col):
    popin_length = end_time - start_time
    time_until_next = None
    if i < len(interval_rows) - 1:
        next_start_idx = int(interval_rows.at[i + 1, start_col])
        next_start_time = df.at[next_start_idx, time_col]
        time_until_next = next_start_time - end_time

    return {
        "popin_length": popin_length,
        "time_until_next": time_until_next,
        "avg_time_during": df[(df[time_col] >= start_time) & (df[time_col] <= end_time)][time_col].mean()
    }

def _compute_precursor_stats(before, time_col, load_col):
    def slope_or_none(subset):
        if len(subset) > 1:
            return linregress(subset[time_col], subset[load_col]).slope
        return None

    return {
        "avg_dload_before": before["dLoad"].mean() if not before.empty else None,
        "slope_before": slope_or_none(before)
    }

def _compute_shape_stats(df, start_idx, end_idx, during, time_col, depth_col):
    t_vals = during[time_col].values
    h_vals = during[depth_col].values

    if len(h_vals) > 2:
        d_depth = np.diff(h_vals)
        d_time = np.diff(t_vals)
        avg_velocity = np.mean(d_depth / d_time)
        curvature = np.gradient(np.gradient(h_vals, t_vals), t_vals)
        avg_curvature = np.mean(curvature)
    else:
        avg_velocity = None
        avg_curvature = None

    return {
        "depth_jump": df.at[end_idx, depth_col] - df.at[start_idx, depth_col],
        "avg_depth_during": during[depth_col].mean() if not during.empty else None,
        "avg_depth_velocity": avg_velocity,
        "avg_curvature_depth": avg_curvature
    }

def calculate_popin_statistics(
    df,
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
    Compute descriptive statistics for each detected pop-in.

    This function calculates time-based, precursor-based, and shape-based features
    for each interval where a pop-in occurred (based on start and end index).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with indentation data and interval metadata.
    precursor_stats : bool
        Whether to calculate average dLoad and slope before the pop-in.
    temporal_stats : bool
        Whether to calculate duration and inter-event timing features.
    popin_shape_stats : bool
        Whether to compute shape-based features like velocity and curvature.
    time_col, load_col, depth_col : str
        Column names for time, load, and depth.
    start_col, end_col : str
        Column names for the start and end index of pop-in intervals.
    before_window, after_window : float
        Time window in seconds to use for context before/after the pop-in.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with per-pop-in statistics added (NaNs elsewhere).
    """
    df = df.copy()
    interval_rows = df.dropna(subset=[start_col, end_col]).copy().reset_index(drop=True)
    df["dLoad"] = df[load_col].diff() / df[time_col].diff()
    results = []

    for i, row in interval_rows.iterrows():
        start_idx = int(row[start_col])
        end_idx = int(row[end_col])
        start_time = df.at[start_idx, time_col]
        end_time = df.at[end_idx, time_col]
        before = df[(df[time_col] >= start_time - before_window) & (df[time_col] < start_time)]
        during = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]
        record = {"start_idx": start_idx, "end_idx": end_idx}

        if temporal_stats:
            record.update(_compute_temporal_stats(start_time, end_time, interval_rows, i, df, time_col, start_col))
        if precursor_stats:
            record.update(_compute_precursor_stats(before, time_col, load_col))
        if popin_shape_stats:
            record.update(_compute_shape_stats(df, start_idx, end_idx, during, time_col, depth_col))

        results.append(record)

    stats_df = pd.DataFrame(results)
    for col in stats_df.columns:
        if col not in [start_col, end_col]:
            df[col] = df[start_col].map(stats_df.set_index("start_idx")[col])

    logger.info(f"Computed pop-in statistics for {len(stats_df)} pop-ins")
    return df

def calculate_curve_summary(df, start_col="start_idx", end_col="end_idx", time_col="Time (s)"):
    """
    Compute curve-level summary statistics about pop-in activity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that includes pop-in intervals.
    start_col, end_col : str
        Column names for start and end indices of pop-ins.
    time_col : str
        Column name for time.

    Returns
    -------
    pd.Series
        Summary metrics: count, total duration, first/last timing, average interval.
    """
    interval_rows = df.dropna(subset=[start_col, end_col]).copy().reset_index(drop=True)
    n_popins = len(interval_rows)
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

    return pd.Series({
        "n_popins": n_popins,
        "total_test_duration": df[time_col].max() - df[time_col].min(),
        "total_popin_duration": total_popin_duration,
        "first_popin_time": first_popin_time,
        "last_popin_time": last_popin_time,
        "avg_time_between_popins": avg_time_between
    })

def default_statistics(df_locate, popin_flag_column="popin", before_window=0.5, after_window=0.5):
    """
    Pipeline to compute pop-in statistics from raw detection flags.

    This function extracts only required columns, selects valid pop-in candidates,
    filters for local maxima, extracts intervals, and calculates descriptive features.

    Parameters
    ----------
    df_locate : pd.DataFrame
        Input data including pop-in candidate flags and indentation curve.
    popin_flag_column : str
        Column name indicating Boolean pop-in candidate (True/False).
    before_window, after_window : float
        Time windows (in seconds) for computing features before and after pop-in.

    Returns
    -------
    pd.DataFrame
        DataFrame with annotated pop-in intervals and computed statistics.
    """
    required_cols = ["Time (s)", "Load (µN)", "Depth (nm)", popin_flag_column]
    if "contact_point" in df_locate.columns:
        required_cols.append("contact_point")
    df_locate = df_locate[required_cols].copy()
    df1 = postprocess_popins_local_max(df_locate, popin_flag_column=popin_flag_column)
    df2 = extract_popin_intervals(df1)
    return calculate_popin_statistics(df2, time_col="Time (s)")


def calculate_stress_strain(df,
                            depth_col="Depth (nm)",
                            load_col="Load (µN)",
                            Reff_um=5.323,
                            min_load_uN=2000,
                            smooth_stress=True,
                            smooth_window=11,
                            smooth_polyorder=2,
                            copy_popin_cols=True):
    """
    Convert load–depth data to stress–strain using Dao et al. (2008) formulas,
    optionally copying pop-in markers from the input DataFrame using timestamp-based matching.

    Parameters
    ----------
    df : pd.DataFrame
        Data with pop-in detection metadata.
    depth_col : str
        Column name for depth in nanometres.
    load_col : str
        Column name for load in microNewtons.
    Reff_um : float
        Effective tip radius in microns.
    min_load_uN : float
        Minimum load filter in microNewtons.
    smooth_stress : bool
        Whether to apply Savitzky-Golay smoothing.
    smooth_window : int
        Window size for smoothing.
    smooth_polyorder : int
        Polynomial order for smoothing.
    copy_popin_cols : bool
        Whether to propagate pop-in metadata by time matching.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with stress/strain (and optionally pop-in info).
    """
    df_filtered = df[df[load_col] >= min_load_uN].copy()
    if df_filtered.empty:
        raise ValueError("No data points remain after filtering by min_load_uN")

    h_m = df_filtered[depth_col] * 1e-9
    P_N = df_filtered[load_col] * 1e-6
    Reff_m = Reff_um * 1e-6

    a = np.sqrt(Reff_m * h_m)
    stress = P_N / (np.pi * a**2)
    strain = h_m / (2.4 * a)

    df_filtered["a_contact_m"] = a
    df_filtered["strain"] = strain
    df_filtered["stress"] = stress / 1e6  # MPa

    if smooth_stress and len(df_filtered) >= smooth_window:
        df_filtered["stress"] = savgol_filter(df_filtered["stress"], smooth_window, smooth_polyorder)

    if copy_popin_cols:
        if "start_idx" in df.columns and "end_idx" in df.columns:
            try:
                feature_stats = df[df["start_idx"].notna()].copy()
                start_times = df.loc[feature_stats["start_idx"].astype(int), "Time (s)"].values
                end_times = df.loc[feature_stats["end_idx"].astype(int), "Time (s)"].values

                df_filtered["popin_start"] = df_filtered["Time (s)"].isin(start_times)
                df_filtered["popin_end"] = df_filtered["Time (s)"].isin(end_times)

                if "popin_selected" in df.columns:
                    popin_times = df[df["popin_selected"]]["Time (s)"].values
                    df_filtered["popin_selected"] = df_filtered["Time (s)"].isin(popin_times)
            except Exception as e:
                logger.warning(f"Could not align pop-in times: {e}")

    logger.info(f"Computed stress–strain for {len(df_filtered)} points")
    return df_filtered

__all__ = [
    "postprocess_popins_local_max",
    "extract_popin_intervals",
    "calculate_popin_statistics",
    "calculate_curve_summary",
    "calculate_stress_strain",
    "default_statistics"
]
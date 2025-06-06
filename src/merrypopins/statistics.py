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


def postprocess_popins_local_max(df, method_column="popin", window=1):
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
    popin_flags = df[method_column] == True
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
    compute_basic_stats=True,
    compute_popin_features=True,
    time_col="Time (s)",
    load_col="Load (µN)",
    depth_col="Depth (nm)",
    start_col="start_idx",
    end_col="end_idx",
    before_window=0.2,
    after_window=0.2
):
    """
    Compute basic and/or feature statistics for pop-in intervals.

    Parameters
    ----------
    df : pd.DataFrame
        Data with start/end indices of pop-ins.
    compute_basic_stats : bool
        Whether to compute duration and time-between-popins.
    compute_popin_features : bool
        Whether to compute slopes and depth-based features.
    time_col : str
        Column name for time.
    load_col : str
        Column name for load.
    depth_col : str
        Column name for depth.
    start_col : str
        Column name for pop-in start indices.
    end_col : str
        Column name for pop-in end indices.
    before_window : float
        Time window before pop-in (s).
    after_window : float
        Time window after pop-in (s).

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns.
    """
    df = df.copy()
    interval_rows = df.dropna(subset=[start_col, end_col]).copy().reset_index(drop=True)
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

        if compute_basic_stats:
            popin_length = end_time - start_time
            time_until_next = None
            if i < len(interval_rows) - 1:
                next_start_idx = int(interval_rows.at[i + 1, start_col])
                next_start_time = df.at[next_start_idx, time_col]
                time_until_next = next_start_time - end_time

            record.update({
                "popin_length": popin_length,
                "time_until_next": time_until_next
            })

        if compute_popin_features:
            dload = df[load_col].diff() / df[time_col].diff()
            df["dLoad"] = dload

            before = df[(df[time_col] >= start_time - before_window) & (df[time_col] < start_time)]
            during = df[(df[time_col] >= start_time) & (df[time_col] <= end_time)]
            after = df[(df[time_col] > end_time) & (df[time_col] <= end_time + after_window)]

            def slope_or_none(subset):
                if len(subset) > 1:
                    return linregress(subset[time_col], subset[load_col]).slope
                return None

            record.update({
                "avg_dload_before": before["dLoad"].mean() if not before.empty else None,
                "avg_dload_after": after["dLoad"].mean() if not after.empty else None,
                "slope_before": slope_or_none(before),
                "slope_after": slope_or_none(after),
                "depth_drop": df.at[end_idx, depth_col] - df.at[start_idx, depth_col],
                "avg_depth_during": during[depth_col].mean() if not during.empty else None,
                "avg_time_during": during[time_col].mean() if not during.empty else None
            })

        results.append(record)

    stats_df = pd.DataFrame(results)
    for col in stats_df.columns:
        if col not in [start_col, end_col]:
            df[col] = df[start_col].map(stats_df.set_index("start_idx")[col])

    logger.info(f"Computed pop-in statistics for {len(stats_df)} pop-ins")
    return df


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
        Original unfiltered data with pop-in detection metadata.
    copy_popin_cols : bool
        If True, uses 'Time (s)' to transfer pop-in info to the filtered result.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with stress/strain (and optionally pop-in info).
    """
    # Step 1: Compute stress–strain
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

    # Step 2: Timestamp-based pop-in transfer
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



def default_statistics(df_locate, method_column="popin", before_window=0.2, after_window=0.2):
    """
    Pipeline for pop-in statistics from load–depth domain.

    Parameters
    ----------
    df_locate : pd.DataFrame
        With pop-in candidate flags and load/time/depth.
    method_column : str
        Column name indicating raw pop-in detection.
    before_window : float
        Time window before pop-in (not yet used here).
    after_window : float
        Time window after pop-in (not yet used here).

    Returns
    -------
    pd.DataFrame
        Copy with popin intervals and basic stats.
    """
    df1 = postprocess_popins_local_max(df_locate, method_column=method_column)
    df2 = extract_popin_intervals(df1)
    return calculate_popin_statistics(df2, time_col="Time (s)")


__all__ = [
    "postprocess_popins_local_max",
    "extract_popin_intervals",
    "calculate_popin_statistics",
    "calculate_stress_strain",
    "default_statistics"
]

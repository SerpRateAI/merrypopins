import pytest
import pandas as pd
import numpy as np
import logging

from merrypopins.statistics import (
    postprocess_popins_local_max,
    extract_popin_intervals,
    calculate_popin_statistics,
    calculate_curve_summary,
    default_statistics,
)

# ========== Fixtures ==========

@pytest.fixture
def synthetic_df_with_popin():
    """Synthetic curve with a single pop-in peak at index 50."""
    time = np.linspace(0, 1, 100)
    load = np.linspace(0, 100, 100)
    depth = np.linspace(0, 50, 100)
    depth[50:52] += 10  # Simulated depth jump

    df = pd.DataFrame({
        "Time (s)": time,
        "Load (µN)": load,
        "Depth (nm)": depth,
    })
    df["popin"] = False
    df.loc[50, "popin"] = True  # Manually flag one pop-in
    return df

@pytest.fixture
def df_no_popin():
    """Curve with no pop-in event flagged."""
    time = np.linspace(0, 1, 100)
    load = np.linspace(0, 100, 100)
    depth = np.linspace(0, 50, 100)
    df = pd.DataFrame({
        "Time (s)": time,
        "Load (µN)": load,
        "Depth (nm)": depth,
        "popin": False,
    })
    return df

# ========== Tests for postprocess_popins_local_max ==========

def test_postprocess_popins_selects_peak(synthetic_df_with_popin, caplog):
    caplog.set_level(logging.INFO)
    df = postprocess_popins_local_max(synthetic_df_with_popin)
    assert "popin_selected" in df.columns
    assert df["popin_selected"].sum() == 1
    assert "local max" in caplog.text

def test_postprocess_popins_none_selected(df_no_popin, caplog):
    caplog.set_level(logging.INFO)
    df = postprocess_popins_local_max(df_no_popin)
    assert "popin_selected" in df.columns
    assert df["popin_selected"].sum() == 0

# ========== Tests for extract_popin_intervals ==========

def test_extract_popin_intervals_has_columns(synthetic_df_with_popin):
    df1 = postprocess_popins_local_max(synthetic_df_with_popin)
    df2 = extract_popin_intervals(df1)
    assert "start_idx" in df2.columns
    assert "end_idx" in df2.columns
    assert df2["start_idx"].notna().sum() >= 0

# ========== Tests for calculate_popin_statistics ==========

def test_calculate_popin_statistics_creates_columns(synthetic_df_with_popin):
    df1 = postprocess_popins_local_max(synthetic_df_with_popin)
    df2 = extract_popin_intervals(df1)
    df3 = calculate_popin_statistics(df2)
    assert "depth_jump" in df3.columns
    assert "popin_length" in df3.columns

def test_calculate_popin_statistics_no_popins(df_no_popin):
    df1 = postprocess_popins_local_max(df_no_popin)
    df2 = extract_popin_intervals(df1)
    df3 = calculate_popin_statistics(df2)
    assert "depth_jump" in df3.columns  # Should still exist
    assert df3["depth_jump"].isna().all()  # But be NaN everywhere

# ========== Tests for calculate_curve_summary ==========

def test_curve_summary_values(synthetic_df_with_popin):
    df1 = postprocess_popins_local_max(synthetic_df_with_popin)
    df2 = extract_popin_intervals(df1)
    summary = calculate_curve_summary(df2)
    assert summary["n_popins"] >= 0
    assert "first_popin_time" in summary

# ========== Tests for default_statistics (full pipeline) ==========

def test_default_statistics_pipeline_runs(synthetic_df_with_popin):
    result = default_statistics(synthetic_df_with_popin)
    assert isinstance(result, pd.DataFrame)
    assert "start_idx" in result.columns
    assert "depth_jump" in result.columns

import pytest
import pandas as pd
import numpy as np
import logging

from merrypopins.statistics import (
    postprocess_popins_local_max,
    extract_popin_intervals,
    calculate_popin_statistics,
    calculate_curve_summary,
    calculate_stress_strain,
    calculate_stress_strain_statistics,
    default_statistics,
    default_statistics_stress_strain,
)

# ========== Fixtures ==========


@pytest.fixture
def realistic_synthetic_df():
    # Mimics structure of df_locate_example.csv
    time = np.linspace(0, 10, 100)
    load = np.linspace(0, 5000, 100)
    depth = np.linspace(0, 1000, 100)
    df = pd.DataFrame(
        {
            "Time (s)": time,
            "Load (µN)": load,
            "Depth (nm)": depth,
            "popin": [False] * 100,
            "popin_confident": [False] * 100,
            "popin_methods": [None] * 100,
            "popin_score": [0] * 100,
        }
    )
    df.loc[50, "popin"] = True
    df.loc[50, "popin_confident"] = True
    df.loc[50, "popin_score"] = 1.0
    df.loc[50, "popin_methods"] = "local_max"
    return df


# ========== Tests for postprocess_popins_local_max ==========


def test_postprocess_popins_selects_peak(realistic_synthetic_df, caplog):
    caplog.set_level(logging.INFO)
    df = realistic_synthetic_df.copy()
    df_result = postprocess_popins_local_max(df)
    assert "popin_selected" in df_result.columns
    assert df_result["popin_selected"].sum() == 1
    assert "local max" in caplog.text


def test_postprocess_popins_none_selected(realistic_synthetic_df):
    df = realistic_synthetic_df.copy()
    df["Load (µN)"] = np.linspace(0, 4000, 100)
    df["popin"] = False
    df["popin_confident"] = False
    df["popin_methods"] = None
    df["popin_score"] = 0
    df_result = postprocess_popins_local_max(df)
    assert "popin_selected" in df_result.columns
    assert df_result["popin_selected"].sum() == 0


# ========== Tests for extract_popin_intervals ==========


def test_extract_popin_intervals_has_columns(realistic_synthetic_df):
    df1 = postprocess_popins_local_max(realistic_synthetic_df.copy())
    df2 = extract_popin_intervals(df1)
    assert "start_idx" in df2.columns
    assert "end_idx" in df2.columns


# ========== Tests for calculate_popin_statistics ==========


def test_calculate_popin_statistics_creates_columns(realistic_synthetic_df):
    df1 = postprocess_popins_local_max(realistic_synthetic_df.copy())
    df2 = extract_popin_intervals(df1)
    df3 = calculate_popin_statistics(df2)
    assert "depth_jump" in df3.columns
    assert "popin_length" in df3.columns


def test_calculate_popin_statistics_no_popins(realistic_synthetic_df):
    df = realistic_synthetic_df.copy()
    df["popin"] = False
    df["popin_confident"] = False
    df["popin_methods"] = None
    df["popin_score"] = 0
    df1 = postprocess_popins_local_max(df)
    df2 = extract_popin_intervals(df1)
    df3 = calculate_popin_statistics(df2)
    assert "depth_jump" in df3.columns
    assert df3["depth_jump"].isna().all()


# ========== Tests for calculate_curve_summary ==========


def test_curve_summary_values(realistic_synthetic_df):
    df1 = postprocess_popins_local_max(realistic_synthetic_df.copy())
    df2 = extract_popin_intervals(df1)
    summary = calculate_curve_summary(df2)
    assert summary["n_popins"] >= 0
    assert "first_popin_time" in summary


# ========== Tests for default_statistics (full pipeline) ==========


def test_default_statistics_pipeline_runs(realistic_synthetic_df):
    result = default_statistics(realistic_synthetic_df.copy())
    assert isinstance(result, pd.DataFrame)
    assert "start_idx" in result.columns
    assert "depth_jump" in result.columns


# ========== Tests for calculate_stress_strain ==========


def test_calculate_stress_strain_runs(realistic_synthetic_df):
    df_stats = default_statistics(realistic_synthetic_df.copy())
    df_strain = calculate_stress_strain(df_stats)
    assert "stress" in df_strain.columns
    assert "strain" in df_strain.columns
    assert not df_strain.empty


# ========== Tests for calculate_stress_strain_statistics ==========


def test_stress_strain_statistics_adds_columns(realistic_synthetic_df):
    df_stats = default_statistics(realistic_synthetic_df.copy())
    df_stress = calculate_stress_strain(df_stats)
    df_final = calculate_stress_strain_statistics(df_stress)
    assert "stress_jump" in df_final.columns
    assert "strain_jump" in df_final.columns


# ========== Tests for default_statistics_stress_strain ==========


def test_default_statistics_stress_strain_pipeline(realistic_synthetic_df):
    df_result = default_statistics_stress_strain(
        realistic_synthetic_df.copy(),
        min_load_uN=10,
        before_window=0.1,
    )
    assert isinstance(df_result, pd.DataFrame), "Result is not a DataFrame"
    assert not df_result.empty, "df_result is empty"
    assert (
        "strain" in df_result.columns
    ), f"'strain' column missing: {df_result.columns}"
    assert (
        "stress" in df_result.columns
    ), f"'stress' column missing: {df_result.columns}"

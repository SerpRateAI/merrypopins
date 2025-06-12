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
def synthetic_df_with_popin():
    time = np.linspace(0, 1, 100)
    load = np.linspace(0, 100, 100)
    load[49] = 90
    load[50] = 110  # Now a local max!
    load[51] = 100
    depth = np.linspace(0, 50, 100)
    depth[50:52] += 10  # Simulated pop-in
    df = pd.DataFrame(
        {
            "Time (s)": time,
            "Load (µN)": load,
            "Depth (nm)": depth,
        }
    )
    df["popin"] = False
    df.loc[50, "popin"] = True
    return df


@pytest.fixture
def df_no_popin():
    time = np.linspace(0, 1, 100)
    load = np.linspace(0, 100, 100)
    depth = np.linspace(0, 50, 100)
    df = pd.DataFrame(
        {
            "Time (s)": time,
            "Load (µN)": load,
            "Depth (nm)": depth,
            "popin": False,
        }
    )
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
    assert "depth_jump" in df3.columns
    assert df3["depth_jump"].isna().all()


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


# ========== Tests for calculate_stress_strain ==========


def test_calculate_stress_strain_runs(synthetic_df_with_popin):
    df_stats = default_statistics(synthetic_df_with_popin)
    df_strain = calculate_stress_strain(df_stats)
    assert "stress" in df_strain.columns
    assert "strain" in df_strain.columns
    assert not df_strain.empty


# ========== Tests for calculate_stress_strain_statistics ==========


def test_stress_strain_statistics_adds_columns(synthetic_df_with_popin):
    df_stats = default_statistics(synthetic_df_with_popin)
    df_stress = calculate_stress_strain(df_stats)
    df_final = calculate_stress_strain_statistics(df_stress)
    assert "stress_jump" in df_final.columns
    assert "strain_jump" in df_final.columns


# ========== Tests for default_statistics_stress_strain ==========


def test_default_statistics_stress_strain_pipeline(synthetic_df_with_popin):
    df_result = default_statistics_stress_strain(
        synthetic_df_with_popin,
        min_load_uN=10,  # Should retain the pop-in
        before_window=0.1,  # Matches your short time span
    )

    # Debug print to inspect what's going on
    print("\n--- df_result.head() ---")
    print(df_result.head())

    print("\n--- Pop-ins after processing ---")
    print(
        df_result["popin_selected"][
            ["Time (s)", "Load (µN)", "stress", "strain", "start_idx", "end_idx"]
        ]
    )

    # Actual test assertions
    assert isinstance(df_result, pd.DataFrame), "Result is not a DataFrame"
    assert not df_result.empty, "df_result is empty"
    assert (
        "strain" in df_result.columns
    ), f"'strain' column missing: {df_result.columns}"
    assert (
        "stress" in df_result.columns
    ), f"'stress' column missing: {df_result.columns}"

import pytest
import pandas as pd
import logging
from pathlib import Path

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
def realistic_df():
    data_path = Path(__file__).parent / "data" / "df_locate_example.csv"
    return pd.read_csv(data_path)


# ========== Tests for postprocess_popins_local_max ==========


def test_postprocess_popins_selects_peak(realistic_df, caplog):
    caplog.set_level(logging.INFO)
    df = postprocess_popins_local_max(realistic_df.copy())
    assert "popin_selected" in df.columns
    assert df["popin_selected"].sum() > 0
    assert "local max" in caplog.text


def test_postprocess_popins_none_selected(realistic_df):
    df = realistic_df.copy()
    df["popin"] = False
    df["popin_confident"] = False
    df["popin_methods"] = None
    df["popin_score"] = 0
    df_result = postprocess_popins_local_max(df)
    assert "popin_selected" in df_result.columns
    assert df_result["popin_selected"].sum() == 0


# ========== Tests for extract_popin_intervals ==========


def test_extract_popin_intervals_has_columns(realistic_df):
    df1 = postprocess_popins_local_max(realistic_df.copy())
    df2 = extract_popin_intervals(df1)
    assert "start_idx" in df2.columns
    assert "end_idx" in df2.columns


# ========== Tests for calculate_popin_statistics ==========


def test_calculate_popin_statistics_creates_columns(realistic_df):
    df1 = postprocess_popins_local_max(realistic_df.copy())
    df2 = extract_popin_intervals(df1)
    df3 = calculate_popin_statistics(df2)
    assert "depth_jump" in df3.columns
    assert "popin_length" in df3.columns


def test_calculate_popin_statistics_handles_empty(realistic_df):
    """
    Ensure the pop-in statistics pipeline runs without crashing when there are no detected pop-ins.
    """
    df = realistic_df.copy()
    df["popin"] = False
    df["popin_confident"] = False
    df["popin_methods"] = None
    df["popin_score"] = 0

    try:
        df1 = postprocess_popins_local_max(df)
        df2 = extract_popin_intervals(df1)
        calculate_popin_statistics(df2)
    except Exception as e:
        pytest.fail(f"Function should not fail on empty pop-in set: {e}")


# ========== Tests for calculate_curve_summary ==========


def test_curve_summary_values(realistic_df):
    df1 = postprocess_popins_local_max(realistic_df.copy())
    df2 = extract_popin_intervals(df1)
    summary = calculate_curve_summary(df2)
    assert summary["n_popins"] >= 0
    assert "first_popin_time" in summary


# ========== Tests for default_statistics (full pipeline) ==========


def test_default_statistics_pipeline_runs(realistic_df):
    result = default_statistics(realistic_df.copy())
    assert isinstance(result, pd.DataFrame)
    assert "start_idx" in result.columns
    assert "depth_jump" in result.columns


# ========== Tests for calculate_stress_strain ==========


def test_calculate_stress_strain_runs(realistic_df):
    df_stats = default_statistics(realistic_df.copy())
    df_strain = calculate_stress_strain(df_stats, min_load_uN=2000)
    assert "stress" in df_strain.columns
    assert "strain" in df_strain.columns
    assert not df_strain.empty


# ========== Tests for calculate_stress_strain_statistics ==========


def test_stress_strain_statistics_adds_columns(realistic_df):
    df_stats = default_statistics(realistic_df.copy())
    df_stress = calculate_stress_strain(df_stats, min_load_uN=2000)
    df_final = calculate_stress_strain_statistics(df_stress)
    assert "stress_jump" in df_final.columns
    assert "strain_jump" in df_final.columns


# ========== Tests for default_statistics_stress_strain ==========


def test_default_statistics_stress_strain_pipeline(realistic_df):
    df_result = default_statistics_stress_strain(
        realistic_df.copy(), min_load_uN=2000, before_window=0.5
    )
    assert isinstance(df_result, pd.DataFrame)
    assert not df_result.empty
    assert "strain" in df_result.columns
    assert "stress" in df_result.columns

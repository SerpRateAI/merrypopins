import pytest
import pandas as pd
import numpy as np
import logging
from pathlib import Path

from merrypopins.statistics import (
    _compute_precursor_stats,
    _compute_shape_stats,
    _compute_stress_strain_precursor_stats,
    _compute_stress_strain_shape_stats,
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


# ----------------------------------------------------------------------
# Unhappy Path Tests
# ----------------------------------------------------------------------


def test_slope_or_none_with_empty_subset():
    """
    Test that None is returned when the subset is empty.
    """
    # Creating an empty DataFrame
    subset = pd.DataFrame({"Time (s)": [], "Load (µN)": []})

    # Run the function
    result = _compute_precursor_stats(subset, "Time (s)", "Load (µN)")["slope_before"]

    # Check if the result is None
    assert result is None


def test_compute_shape_stats_with_few_points():
    """
    Test that avg_velocity and avg_curvature are set to None when there are not enough data points (<= 2).
    """
    # Creating a mock DataFrame with only 1 point for depth values (h_vals) during pop-in
    subset = pd.DataFrame(
        {
            "Time (s)": [1, 2],
            "Depth (nm)": [100, 101],  # Two points should trigger the else case
        }
    )

    # Run the function
    result = _compute_shape_stats(subset, 0, 1, subset, "Time (s)", "Depth (nm)")

    # Check if avg_velocity and avg_curvature are None when there are less than 3 depth points
    assert result["avg_depth_velocity"] is None
    assert result["avg_curvature_depth"] is None


def test_compute_shape_stats_with_multiple_points():
    """
    Test that avg_velocity and avg_curvature are computed when there are more than 2 points in the data.
    """
    # Creating a mock DataFrame with more than 2 points
    subset = pd.DataFrame(
        {
            "Time (s)": [1, 2, 3, 4],
            "Depth (nm)": [100, 102, 105, 110],  # More than 2 points
        }
    )

    # Run the function
    result = _compute_shape_stats(subset, 0, 3, subset, "Time (s)", "Depth (nm)")

    # Check that avg_velocity and avg_curvature are not None
    assert result["avg_depth_velocity"] is not None
    assert result["avg_curvature_depth"] is not None


def test_calculate_curve_summary_no_popins(realistic_df):
    """
    Test case for the `calculate_curve_summary` function when no pop-ins are detected in the data.

    This test ensures that the function properly handles the case where there are no pop-ins
    and returns the expected values for `total_popin_duration`, `avg_time_between_popins`,
    `first_popin_time`, and `last_popin_time`, which should all be set to `NaN` or `0.0`.

    Args:
        realistic_df (pd.DataFrame): The test DataFrame provided by the fixture.
    """
    # Prepare a DataFrame with no pop-ins
    df_no_popins = realistic_df.copy()
    df_no_popins["popin"] = False  # Mark all rows as not pop-ins

    # Add default start_idx and end_idx columns to avoid the KeyError
    df_no_popins["start_idx"] = None
    df_no_popins["end_idx"] = None

    # Run the function
    summary = calculate_curve_summary(df_no_popins)

    # Assert that the returned summary has the expected values
    assert summary["n_popins"] == 0  # No pop-ins detected
    assert summary["total_popin_duration"] == 0.0  # Duration should be 0
    assert np.isnan(summary["avg_time_between_popins"])  # No intervals, so NaN
    assert np.isnan(summary["first_popin_time"])  # No pop-ins, so NaN
    assert np.isnan(summary["last_popin_time"])  # No pop-ins, so NaN


def test_calculate_stress_strain_missing_columns(realistic_df):
    """
    Test case for the `calculate_stress_strain` function when required columns are missing.

    This test ensures that the function raises a `ValueError` when one or more of the
    required columns ('Depth (nm)', 'Load (µN)', 'Time (s)') are missing from the input DataFrame.

    Args:
        realistic_df (pd.DataFrame): The test DataFrame provided by the fixture.
    """
    # Prepare the DataFrame with a missing column (e.g., "Depth (nm)")
    df_missing_column = realistic_df.copy()
    df_missing_column.drop(
        columns=["Depth (nm)"], inplace=True
    )  # Drop the required column

    # Run the function and assert that it raises a ValueError
    with pytest.raises(ValueError):
        calculate_stress_strain(df_missing_column)


def test_calculate_stress_strain_empty_after_filtering(realistic_df):
    """
    Test case for the `calculate_stress_strain` function when no data remains after applying the min_load_uN filter.

    This test ensures that the function raises a `ValueError` when the DataFrame becomes empty after filtering
    based on the minimum load threshold.

    Args:
        realistic_df (pd.DataFrame): The test DataFrame provided by the fixture.
    """
    # Prepare the DataFrame with all values below the min_load_uN threshold
    df_filtered_empty = realistic_df.copy()
    df_filtered_empty["Load (µN)"] = 0  # Set all loads to 0

    # Run the function and assert that it raises a ValueError
    with pytest.raises(ValueError):
        calculate_stress_strain(
            df_filtered_empty, min_load_uN=2000
        )  # Threshold set higher than any data


def test_compute_stress_strain_shape_stats_during_len_less_than_3(realistic_df):
    """
    Test case for `_compute_stress_strain_shape_stats` when the length of 'during' is less than 3.

    This test ensures that the function correctly returns None for stress and strain statistics
    when the length of the 'during' period is too short.

    Args:
        realistic_df (pd.DataFrame): The test DataFrame provided by the fixture.
    """
    # Prepare a DataFrame where the 'during' period is shorter than 3 points
    df_short_during = realistic_df.copy().iloc[
        :2
    ]  # Only two points, making the during period too short
    during = df_short_during  # The `during` period is the selected short DataFrame

    # Call the function with the correct arguments (4 in total)
    result = _compute_stress_strain_shape_stats(
        during, "Time (s)", "Load (µN)", "Depth (nm)"
    )

    # Assert that None values are returned for the short period
    assert result["avg_stress_during"] is None
    assert result["avg_strain_during"] is None
    assert result["stress_slope"] is None
    assert result["strain_slope"] is None


def test_compute_stress_strain_precursor_stats_len_less_than_2(realistic_df):
    """
    Test case for `_compute_stress_strain_precursor_stats` when the length of the input data is less than 2.

    This test ensures that the function correctly returns None for slope calculation when there are not enough data points.

    Args:
        realistic_df (pd.DataFrame): The test DataFrame provided by the fixture.
    """
    # Prepare a DataFrame with only one point for the 'before' period
    df_one_point = realistic_df.copy().head(1)
    before = df_one_point[["Time (s)", "Load (µN)"]].copy()

    # Add a dummy strain column (since it's required by the function)
    before.loc[:, "strain"] = 0.0

    result = _compute_stress_strain_precursor_stats(
        before, "Time (s)", "Load (µN)", "strain"
    )

    # Assert that the slope values are None due to only one data point
    assert result["stress_slope_before"] is None
    assert result["strain_slope_before"] is None


def test_compute_stress_strain_precursor_stats_len_greater_than_1(realistic_df):
    """
    Test case for `_compute_stress_strain_precursor_stats` when the length of the input data is greater than 1.

    This test ensures that the function calculates the slope correctly when there are more than one data point.

    Args:
        realistic_df (pd.DataFrame): The test DataFrame provided by the fixture.
    """
    # Prepare a DataFrame with more than one point for the 'before' period
    df_multiple_points = realistic_df.copy().head(5)
    before = df_multiple_points[["Time (s)", "Load (µN)"]].copy()

    # Add a dummy strain column
    before.loc[:, "strain"] = 0.0

    result = _compute_stress_strain_precursor_stats(
        before, "Time (s)", "Load (µN)", "strain"
    )

    # Assert that the slope values are valid for multiple data points
    assert result["stress_slope_before"] is not None
    assert result["strain_slope_before"] is not None

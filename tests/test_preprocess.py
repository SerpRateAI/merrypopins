import pytest
import pandas as pd
import numpy as np
import logging

from indenter.preprocess import remove_initial_data, rescale_data, preprocess_pipeline

# ========== Fixtures ==========

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Depth (nm)": [-5, -3, -1, 0, 1, 2, 3, 4],
        "Load (µN)": [10, 5, 2, 1, 2, 4, 7, 20]
    })

@pytest.fixture
def no_crossing_df():
    # Data that never rises above threshold
    return pd.DataFrame({
        "Depth (nm)": np.linspace(-5, 2, 50),
        "Load (µN)": np.random.normal(loc=0.5, scale=0.1, size=50)
    })

@pytest.fixture
def short_df():
    return pd.DataFrame({
        "Depth (nm)": [0, 1],
        "Load (µN)": [5, 10]
    })

# ========== Tests for remove_initial_data ==========

def test_remove_initial_data_basic(sample_df, caplog):
    caplog.set_level(logging.INFO)
    result = remove_initial_data(sample_df)
    assert len(result) == 4  # 8 total, min at index 3 → remove up to 4
    assert result.iloc[0]["Load (µN)"] == 2
    assert "Removed first 4 points up to minimum Load" in caplog.text

def test_remove_initial_data_min_at_end(caplog):
    df = pd.DataFrame({
        "Depth (nm)": [1, 2, 3],
        "Load (µN)": [10, 5, 1]  # Min at last index
    })
    caplog.set_level(logging.WARNING)
    result = remove_initial_data(df)
    assert len(result) == 3
    assert "Minimum at end of data" in caplog.text

# ========== Tests for rescale_data ==========

def test_rescale_data_successful(sample_df, caplog):
    caplog.set_level(logging.INFO)
    df_cleaned = sample_df.iloc[4:].reset_index(drop=True)  # Skip initial drop for cleaner test
    rescaled = rescale_data(df_cleaned, N_baseline=3, k=1.0, window_length=5)
    shift = sample_df.iloc[4]["Depth (nm)"]
    assert np.isclose(rescaled["Depth (nm)"].iloc[0], 0.0)
    assert "Auto-rescaled" in caplog.text

def test_rescale_data_no_crossing(no_crossing_df, caplog):
    caplog.set_level(logging.WARNING)
    result = rescale_data(no_crossing_df, N_baseline=10, k=10)
    assert result.equals(no_crossing_df)
    assert "No crossing above auto-threshold" in caplog.text

def test_rescale_data_short_window(short_df):
    # Checks fallback if window_length is too long for dataset
    rescaled = rescale_data(short_df, window_length=11)
    # Should still return without crash, even if no threshold crossed
    assert isinstance(rescaled, pd.DataFrame)

# ========== Tests for preprocess_pipeline ==========

def test_preprocess_pipeline_combined(sample_df):
    processed = preprocess_pipeline(sample_df, N_baseline=3, k=1.0, window_length=5)
    assert isinstance(processed, pd.DataFrame)
    assert processed.shape[0] < sample_df.shape[0]  # At least some points removed

def test_pipeline_no_shift_if_no_crossing(no_crossing_df):
    result = preprocess_pipeline(no_crossing_df, N_baseline=10, k=10)
    assert result.equals(remove_initial_data(no_crossing_df))  # Should only remove initial points


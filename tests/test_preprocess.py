import pytest
import pandas as pd
import numpy as np
import logging

from indenter.preprocess import (
    remove_pre_min_load,
    rescale_data,
    finalise_contact_index,
    default_preprocess
)

# ========== Fixtures ==========

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Depth (nm)": [-5, -3, -1, 0, 1, 2, 3, 4],
        "Load (µN)": [10, 5, 2, 1, 2, 4, 7, 20]
    })

@pytest.fixture
def no_crossing_df():
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

# ========== Tests for remove_pre_min_load ==========

def test_remove_pre_min_load_basic(sample_df, caplog):
    caplog.set_level(logging.INFO)
    result = remove_pre_min_load(sample_df)
    assert len(result) == 4  # 8 total, min at index 3 → remove up to 4
    assert result.iloc[0]["Load (µN)"] == 2
    assert "Removed first 4 points up to minimum Load" in caplog.text

def test_remove_pre_min_load_min_at_end(caplog):
    df = pd.DataFrame({
        "Depth (nm)": [1, 2, 3],
        "Load (µN)": [10, 5, 1]  # Min at last index
    })
    caplog.set_level(logging.WARNING)
    result = remove_pre_min_load(df)
    assert len(result) == 3
    assert "Minimum at end of data" in caplog.text

# ========== Tests for rescale_data ==========

def test_rescale_data_successful(sample_df, caplog):
    caplog.set_level(logging.INFO)
    df_cleaned = sample_df.iloc[4:].reset_index(drop=True)  # Skip initial drop for cleaner test
    rescaled = rescale_data(df_cleaned, N_baseline=3, k=1.0, window_length=5)
    assert np.isclose(rescaled["Depth (nm)"].iloc[0], 0.0)
    assert "Auto-rescaled" in caplog.text

def test_rescale_data_no_crossing(no_crossing_df, caplog):
    caplog.set_level(logging.WARNING)
    result = rescale_data(no_crossing_df, N_baseline=10, k=10)
    assert result.equals(no_crossing_df)
    assert "No crossing above auto-threshold" in caplog.text

def test_rescale_data_short_window(short_df):
    rescaled = rescale_data(short_df, window_length=11)
    assert isinstance(rescaled, pd.DataFrame)

# ========== Tests for finalise_contact_index ==========

def test_finalise_contact_index_flags_and_removes(sample_df):
    df_rescaled = rescale_data(sample_df.iloc[4:].reset_index(drop=True), N_baseline=3, k=1.0)
    result = finalise_contact_index(df_rescaled, remove_pre_contact=True, add_flag_column=True)
    assert "contact_point" in result.columns
    assert result["contact_point"].sum() == 1
    assert result.iloc[0]["Depth (nm)"] == 0.0  # First row should now be the contact point

def test_finalise_contact_index_only_flags(sample_df):
    df_rescaled = rescale_data(sample_df.iloc[4:].reset_index(drop=True), N_baseline=3, k=1.0)
    result = finalise_contact_index(df_rescaled, remove_pre_contact=False, add_flag_column=True)
    assert "contact_point" in result.columns
    assert result["contact_point"].sum() == 1

def test_finalise_contact_index_no_flag_or_trim(sample_df):
    df_rescaled = rescale_data(sample_df.iloc[4:].reset_index(drop=True), N_baseline=3, k=1.0)
    result = finalise_contact_index(df_rescaled, remove_pre_contact=False, add_flag_column=False)
    assert "contact_point" not in result.columns
    assert result.shape[0] == df_rescaled.shape[0]

# ========== Tests for default_preprocess ==========

def test_default_preprocess_combined(sample_df):
    processed = default_preprocess(sample_df)
    assert isinstance(processed, pd.DataFrame)
    assert "contact_point" in processed.columns
    assert processed["contact_point"].sum() == 1

def test_default_preprocess_handles_no_crossing(no_crossing_df):
    result = default_preprocess(no_crossing_df)
    # Should still be a DataFrame, possibly empty if no contact found
    assert isinstance(result, pd.DataFrame)
    assert "contact_point" in result.columns

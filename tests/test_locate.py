import numpy as np
import pandas as pd
import pytest

from merrypopins.locate import (
    compute_stiffness,
    compute_features,
    detect_popins_iforest,
    detect_popins_fd_fourier,
    detect_popins_savgol,
    detect_popins_cnn,
    default_locate,
)


# ----------------------------------------------------------------------
# Fixtures: perfect line vs. line with a single jump
# ----------------------------------------------------------------------
@pytest.fixture
def simple_curve():
    depth = np.linspace(0, 10, 101)
    load = 2 * depth
    return pd.DataFrame({"Depth (nm)": depth, "Load (µN)": load})


@pytest.fixture
def curve_with_jump():
    depth = np.linspace(0, 10, 101)
    load = 2 * depth
    load[50] += 20  # big jump at index 50
    return pd.DataFrame({"Depth (nm)": depth, "Load (µN)": load})


# ----------------------------------------------------------------------
# compute_stiffness / compute_features
# ----------------------------------------------------------------------
def test_compute_stiffness_constant(simple_curve):
    stiff = compute_stiffness(simple_curve, window=5)
    mid = stiff.iloc[10:-10]
    assert np.allclose(mid, 2, atol=1e-6)
    assert np.isnan(stiff.iloc[0]) and np.isnan(stiff.iloc[-1])


def test_compute_features_adds_columns(simple_curve):
    df_feat = compute_features(simple_curve, window=5)
    for col in ("stiffness", "stiff_diff", "curvature"):
        assert col in df_feat.columns
    interior = df_feat.iloc[10:-10]
    assert np.allclose(interior["stiff_diff"], 0, atol=1e-6)
    assert np.allclose(interior["curvature"], 0, atol=1e-6)


# ----------------------------------------------------------------------
# Validation: error handling
# ----------------------------------------------------------------------
def test_stiffness_missing_columns():
    df = pd.DataFrame({"Depth (nm)": [1, 2, 3]})
    with pytest.raises(ValueError, match="Required columns.*Load.*not found"):
        compute_stiffness(df, depth_col="Depth (nm)", load_col="Load (µN)", window=3)


def test_stiffness_too_few_points():
    df = pd.DataFrame({"Depth (nm)": [0.0, 1.0], "Load (µN)": [0.0, 2.0]})
    with pytest.raises(ValueError, match="Not enough data points.*for window size"):
        compute_stiffness(df, window=5)


# ----------------------------------------------------------------------
# detect_popins_fd_fourier
# ----------------------------------------------------------------------
def test_detect_popins_fd_fourier_no_jump(simple_curve):
    df0 = detect_popins_fd_fourier(simple_curve, threshold=3.0)
    n0 = df0["popin_fd"].sum()
    assert n0 <= 3


def test_detect_popins_fd_fourier_with_jump(curve_with_jump):
    df0 = detect_popins_fd_fourier(curve_with_jump, threshold=3.0)
    assert df0["popin_fd"].sum() > 0
    assert df0["popin_fd"].iloc[48:53].any()


# ----------------------------------------------------------------------
# detect_popins_savgol
# ----------------------------------------------------------------------
def test_detect_popins_savgol_no_jump(simple_curve):
    df0 = detect_popins_savgol(
        simple_curve, window_length=11, polyorder=2, threshold=3.0
    )
    assert df0["popin_savgol"].sum() <= 3


def test_detect_popins_savgol_with_jump(curve_with_jump):
    df1 = detect_popins_savgol(
        curve_with_jump, window_length=11, polyorder=2, threshold=1.0
    )
    assert df1["popin_savgol"].sum() > 0
    assert df1["popin_savgol"].iloc[48:53].any()


# ----------------------------------------------------------------------
# detect_popins_iforest
# ----------------------------------------------------------------------
def test_detect_popins_iforest_no_jump(simple_curve):
    df0 = detect_popins_iforest(simple_curve, contamination=0.01, random_state=42)
    assert df0["popin_iforest"].sum() <= 2


def test_detect_popins_iforest_with_jump(curve_with_jump):
    df1 = detect_popins_iforest(curve_with_jump, contamination=0.01, random_state=42)
    assert df1["popin_iforest"].sum() > 0
    assert df1["popin_iforest"].iloc[48:53].any()


# ----------------------------------------------------------------------
# detect_popins_cnn
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_detect_popins_cnn_basic(curve_with_jump):
    df_cnn = detect_popins_cnn(
        curve_with_jump, window_size=20, epochs=2, threshold_multiplier=1.0
    )
    flags = df_cnn["popin_cnn"].sum()
    assert flags > 0
    idxs = np.where(df_cnn["popin_cnn"])[0]
    assert idxs.min() >= 10 and idxs.max() <= len(curve_with_jump) - 10


# ----------------------------------------------------------------------
# default_locate
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_default_locate_combine(curve_with_jump):
    df_all = default_locate(
        curve_with_jump,
        iforest_contamination=0.01,
        cnn_window_size=20,
        cnn_epochs=1,
        cnn_threshold_multiplier=1.0,
        fd_threshold=3.0,
        savgol_window_length=11,
        savgol_polyorder=2,
        savgol_threshold=1.0,
    )
    for col in ("popin_iforest", "popin_cnn", "popin_fd", "popin_savgol", "popin"):
        assert col in df_all.columns

    union_count = df_all["popin"].sum()
    max_single = (
        df_all[["popin_iforest", "popin_cnn", "popin_fd", "popin_savgol"]]
        .sum(axis=1)
        .max()
    )
    assert union_count >= max_single

    assert "popin_methods" in df_all.columns
    assert "popin_score" in df_all.columns
    assert "popin_confident" in df_all.columns
    assert df_all["popin_confident"].dtype == bool

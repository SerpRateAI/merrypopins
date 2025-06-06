import pytest
from pathlib import Path
import pandas as pd
from merrypopins.make_dataset import merrypopins_pipeline

# ----------------------------------------------------------------------
# Fixtures: Load and prepare test data
# ----------------------------------------------------------------------


@pytest.fixture
def real_test_file():
    return Path("tests/data/test_indent.txt")


# ----------------------------------------------------------------------
# Test: Pipeline successfully runs and returns annotated DataFrame
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_pipeline_runs_and_returns_dataframe(real_test_file):
    """
    Test if merrypopins_pipeline executes and returns a valid DataFrame.
    """
    df = merrypopins_pipeline(txt_path=real_test_file)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "popin_confident" in df.columns


# ----------------------------------------------------------------------
# Test: Pipeline creates the expected output plot
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_pipeline_creates_visualization_file(real_test_file, tmp_path):
    """
    Test if a .png file is created after running the pipeline.
    """
    plot_dir = tmp_path / "plots"
    merrypopins_pipeline(txt_path=real_test_file, save_plot_dir=plot_dir)
    expected_plot = plot_dir / f"{real_test_file.stem}.png"
    assert expected_plot.exists()


# ----------------------------------------------------------------------
# Test: Pipeline respects all override settings
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_pipeline_respects_all_settings(real_test_file):
    """
    Test the pipeline's configuration parameters to ensure they are accepted and processed.
    """
    df = merrypopins_pipeline(
        txt_path=real_test_file,
        iforest_contamination=0.01,
        iforest_random_state=123,
        cnn_window_size=32,
        cnn_epochs=2,
        cnn_threshold_multiplier=3.0,
        cnn_batch_size=16,
        cnn_validation_split=0.1,
        fd_threshold=2.0,
        fd_spacing=0.5,
        savgol_window_length=5,
        savgol_polyorder=2,
        savgol_threshold=2.5,
        sg_deriv_order=1,
        stiffness_window=3,
        trim_edges_enabled=True,
        trim_margin=1,
        use_iforest=True,
        use_cnn=True,
        use_fd=True,
        use_savgol=True,
    )
    assert isinstance(df, pd.DataFrame)
    assert "popin_confident" in df.columns

"""
make_dataset.py
---------------

This module provides a function to execute the full Merrypopins pipeline,
from loading the dataset to preprocessing, locating pop-ins, and visualizing the results.

It integrates various methods for pop-in detection and saves the results as a DataFrame.

Provides:
- `merrypopins_pipeline`: A function that orchestrates the entire process.
- Saves visualizations of the detected pop-ins.
- Returns a DataFrame with all annotations.
"""

from pathlib import Path
import matplotlib.pyplot as plt
from merrypopins.load_datasets import load_txt
from merrypopins.preprocess import default_preprocess
from merrypopins.locate import default_locate


def merrypopins_pipeline(
    txt_path,
    iforest_contamination=0.001,
    iforest_random_state=None,
    cnn_window_size=64,
    cnn_epochs=10,
    cnn_threshold_multiplier=5.0,
    cnn_batch_size=32,
    cnn_validation_split=0.0,
    fd_threshold=3.0,
    fd_spacing=1.0,
    savgol_window_length=11,
    savgol_polyorder=2,
    savgol_threshold=3.0,
    sg_deriv_order=1,
    stiffness_window=5,
    trim_edges_enabled=True,
    trim_margin=None,
    max_load_trim_enabled=True,
    use_iforest=True,
    use_cnn=True,
    use_fd=True,
    use_savgol=True,
    depth_col="Depth (nm)",
    load_col="Load (µN)",
    save_plot_dir=Path("visualisations"),
):
    """
    Executes the full Merrypopins pipeline: load -> preprocess -> locate -> visualize.

    Args:
        txt_path (Path or str): Path to the indentation .txt file.
        iforest_contamination (float): Contamination level for IsolationForest.
        iforest_random_state (int): Random seed for IsolationForest.
        cnn_window_size (int): CNN autoencoder window size.
        cnn_epochs (int): CNN training epochs.
        cnn_threshold_multiplier (float): Threshold multiplier for CNN.
        cnn_batch_size (int): Batch size for CNN training.
        cnn_validation_split (float): Fraction of data for validation.
        fd_threshold (float): Threshold for finite difference method.
        fd_spacing (float): Sampling interval for Fourier derivative.
        savgol_window_length (int): Window length for Savitzky-Golay filter.
        savgol_polyorder (int): Polynomial order for Savitzky-Golay.
        savgol_threshold (float): Threshold for Savitzky-Golay.
        sg_deriv_order (int): Derivative order for Savitzky-Golay.
        stiffness_window (int): Smoothing window for stiffness calculation.
        trim_edges_enabled (bool): Whether to trim pop-ins at curve edges.
        trim_margin (int): Margin to trim detections at beginning.
        max_load_trim_enabled (bool): Whether to trim pop-ins based on max load.
        use_iforest (bool): Enable IsolationForest detection.
        use_cnn (bool): Enable CNN detection.
        use_fd (bool): Enable Fourier detection.
        use_savgol (bool): Enable Savitzky-Golay detection.
        depth_col (str): Column name for depth.
        load_col (str): Column name for load.
        save_plot_dir (Path): Directory where plots will be saved.

    Returns:
        DataFrame: Final DataFrame with all annotations.
    """
    txt_path = Path(txt_path)
    df = load_txt(txt_path)
    df_pre = default_preprocess(df)
    df_loc = default_locate(
        df_pre,
        iforest_contamination=iforest_contamination,
        iforest_random_state=iforest_random_state,
        cnn_window_size=cnn_window_size,
        cnn_epochs=cnn_epochs,
        cnn_threshold_multiplier=cnn_threshold_multiplier,
        cnn_batch_size=cnn_batch_size,
        cnn_validation_split=cnn_validation_split,
        fd_threshold=fd_threshold,
        fd_spacing=fd_spacing,
        savgol_window_length=savgol_window_length,
        savgol_polyorder=savgol_polyorder,
        savgol_threshold=savgol_threshold,
        sg_deriv_order=sg_deriv_order,
        stiffness_window=stiffness_window,
        trim_edges_enabled=trim_edges_enabled,
        trim_margin=trim_margin,
        max_load_trim_enabled=max_load_trim_enabled,
        use_iforest=use_iforest,
        use_cnn=use_cnn,
        use_fd=use_fd,
        use_savgol=use_savgol,
        depth_col=depth_col,
        load_col=load_col,
    )

    save_plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = save_plot_dir / f"{txt_path.stem}.png"

    plt.figure(figsize=(8, 6))
    plt.plot(
        df_loc[depth_col],
        df_loc[load_col],
        label="Preprocessed",
        alpha=0.4,
        color="orange",
    )

    colors = {
        "popin_iforest": "red",
        "popin_cnn": "purple",
        "popin_fd": "darkorange",
        "popin_savgol": "green",
    }
    markers = {
        "popin_iforest": "^",
        "popin_cnn": "v",
        "popin_fd": "x",
        "popin_savgol": "D",
    }

    for method, color in colors.items():
        mdf = df_loc[df_loc[method]]
        plt.scatter(
            mdf[depth_col],
            mdf[load_col],
            c=color,
            label=method.replace("popin_", "").capitalize(),
            marker=markers[method],
            alpha=0.7,
        )

    confident = df_loc[df_loc["popin_confident"]]
    plt.scatter(
        confident[depth_col],
        confident[load_col],
        edgecolors="k",
        facecolors="none",
        label="Majority Vote (2+)",
        s=100,
        linewidths=1.5,
    )

    plt.xlabel(depth_col)
    plt.ylabel(load_col)
    plt.title(f"Pop-in Detection: {txt_path.stem}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return df_loc


__all__ = ["merrypopins_pipeline"]

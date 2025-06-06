# Merrypopins

[![merrypopins CI Tests](https://github.com/SerpRateAI/merrypopins/actions/workflows/python-app.yml/badge.svg)](https://github.com/SerpRateAI/merrypopins/actions/workflows/python-app.yml)
[![📘 Merrypopins Documentation](https://img.shields.io/badge/docs-view-blue?logo=readthedocs)](https://serprateai.github.io/merrypopins/)
[![PyPI](https://img.shields.io/pypi/v/merrypopins.svg)](https://pypi.org/project/merrypopins/)
[![Python](https://img.shields.io/pypi/pyversions/merrypopins.svg)](https://pypi.org/project/merrypopins/)
[![codecov](https://codecov.io/gh/SerpRateAI/merrypopins/branch/main/graph/badge.svg)](https://codecov.io/gh/SerpRateAI/merrypopins)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/merrypopins)](https://pepy.tech/project/merrypopins)
[![Issues](https://img.shields.io/github/issues/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/issues)
[![Dependencies](https://img.shields.io/librariesio/github/SerpRateAI/merrypopins)](https://github.com/SerpRateAI/merrypopins/network/dependencies)
[![Last commit](https://img.shields.io/github/last-commit/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/commits/main)
[![Release](https://img.shields.io/github/release-date/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/releases)
[![Contributors](https://img.shields.io/github/contributors/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/graphs/contributors)

**merrypopins** is a Python library to streamline the workflow of nano‑indentation experiment data processing, automated pop-in detection and analysis. It provides five core modules:

- **`load_datasets`**: Load and parse `.txt` measurement files and `.tdm`/`.tdx` metadata files into structured pandas DataFrames. Automatically detects headers, timestamps, and measurement channels.
- **`preprocess`**: Clean and normalize indentation data with filtering, baseline correction, and contact point detection.
- **`locate`**: Identify and extract pop‑in events within indentation curves using advanced detection algorithms, including:
  - Isolation Forest anomaly detection
  - CNN Autoencoder reconstruction error
  - Fourier-based derivative outlier detection
  - Savitzky-Golay smoothed gradient thresholds
- **`statistics`**: Perform statistical analysis and model fitting on located pop‑in events (e.g., frequency, magnitude, distribution).
- **`make_dataset`**: Construct enriched datasets by running the full merrypopins pipeline and exporting annotated results and visualizations.

Merrypopins is developed by [Cahit Acar](mailto:c.acar.business@gmail.com), [Anna Marcelissen](mailto:anna.marcelissen@live.nl), [Hugo van Schrojenstein Lantman](mailto:h.w.vanschrojensteinlantman@uu.nl), and [John M. Aiken](mailto:johnm.aiken@gmail.com).

---

## Installation

```bash
# From PyPI
pip install merrypopins

# For development
git clone https://github.com/SerpRateAI/merrypopins.git
cd merrypopins
pip install -e .
```

merrypopins supports Python 3.10+ and depends on:

- `matplotlib`
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `tensorflow`

These are installed automatically via `pip`.

---

## Quickstart

### Importing merrypopins Modules

```python
from pathlib import Path
from merrypopins.load_datasets import load_txt, load_tdm
from merrypopins.preprocess import default_preprocess, remove_pre_min_load, rescale_data, finalise_contact_index
from merrypopins.locate import default_locate
from merrypopins.make_dataset import merrypopins_pipeline
```

### Load Indentation Data and Metadata

```python
# 1) Load indentation data:
data_file = Path("data/experiment1.txt")
df = load_txt(data_file)
print(df.head())
print("Timestamp:", df.attrs['timestamp'])
print("Number of Points:", df.attrs['num_points'])

# 2) Load tdm metadata:
tdm_meta_file = Path("data/experiment1.tdm")
# Load tdm metadata and channels this will create dataframe for root and channels
df_tdm_meta_root, df_tdm_meta_channels = load_tdm(tdm_meta_file)
# The root metadata is stored as one row with their respective columns
print(df_tdm_meta_root.head())
# To be able to read all the columns of root metadata dataframe it can be transposed
df_tdm_meta_root = df_tdm_meta_root.T.reset_index()
df_tdm_meta_root.columns = ['attribute', 'value']
print(df_tdm_meta_root.head(50))
# The channel metadata is stored as multiple rows with their respective columns
print(df_tdm_meta_channels.head(50))
```

### Preprocess Data

#### Option 1: Use default pipeline

```python
# This applies:
# 1. Removes all rows before minimum Load
# 2. Detects contact point and shifts Depth so contact = 0
# 3. Removes Depth < 0 rows and adds a flag for the contact point

df_processed = default_preprocess(df)

print(df_processed.head())
print("Contact point index:", df_processed[df_processed["contact_point"]].index[0])
```

#### Option 2: Customize each step (with optional arguments)

```python
# Step 1: Remove initial noise based on minimum Load
df_clean = remove_pre_min_load(df, load_col="Load (µN)")

# Step 2: Automatically detect contact point and zero the depth
df_rescaled = rescale_data(
    df_clean,
    depth_col="Depth (nm)",
    load_col="Load (µN)",
    N_baseline=30,     # number of points for baseline noise estimation
    k=5.0,             # noise threshold multiplier
    window_length=7,   # Savitzky-Golay smoothing window (must be odd)
    polyorder=2        # Polynomial order for smoothing
)

# Step 3: Trim rows before contact and/or flag the point
df_final = finalise_contact_index(
    df_rescaled,
    depth_col="Depth (nm)",
    remove_pre_contact=True,       # remove rows where depth < 0
    add_flag_column=True,          # add a boolean column marking the contact point
    flag_column="contact_point"    # customize the column name if needed
)

print(df_final[df_final["contact_point"]])  # display contact row
print("Contact point index:", df_final[df_final["contact_point"]].index[0])
```
🧪 Tip
You can omit or modify any step depending on your data:

- Skip remove_pre_min_load() if your data is already clean.
- Set remove_pre_contact=False if you want to retain all data.
- Customize flag_column to integrate with your own schema.

### Locate Pop-in Events

#### Detect Pop-ins using Default Method

```python
# Detect pop-ins using all methods
results = default_locate(df_processed)
print(results[results.popin])
```

### Customize Detection Thresholds

```python
results_tuned = default_locate(
    df_processed,
    iforest_contamination=0.002,
    cnn_threshold_multiplier=4.0,
    fd_threshold=2.5,
    savgol_threshold=2.0
)
```

### Visualize Detections

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plt.plot(results_tuned["Depth (nm)"], results_tuned["Load (µN)"], label="Preprocessed", alpha=0.4, color='orange')

colors = {
    "popin_iforest": 'red',
    "popin_cnn": 'purple',
    "popin_fd": 'darkorange',
    "popin_savgol": 'green'
}
markers = {
    "popin_iforest": '^',
    "popin_cnn": 'v',
    "popin_fd": 'x',
    "popin_savgol": 'D'
}

for method, color in colors.items():
    mdf = results_tuned[results_tuned[method]]
    plt.scatter(mdf["Depth (nm)"], mdf["Load (µN)"],
                c=color, label=method.replace("popin_", "").capitalize(),
                marker=markers[method], alpha=0.7)

confident = results_tuned[results_tuned["popin_confident"]]
plt.scatter(confident["Depth (nm)"], confident["Load (µN)"],
            edgecolors='k', facecolors='none', label="Majority Vote (2+)", s=100, linewidths=1.5)

plt.xlabel("Depth (nm)"); plt.ylabel("Load (µN)")
plt.title("Pop-in Detections by All Methods")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()
```

### Run Full Pipeline with merrypopins_pipeline

This function runs the entire merrypopins workflow, from loading data to locating pop-ins and generating visualizations.

#### Define Input and Output Paths

```python
# Define the text file that will be processed and output directory that will contain the visualization
text_file = Path("datasets/6microntip_slowloading/grain9_6um_indent03_HL_QS_LC.txt")
output_dir = Path("visualisations/6microntip_slowloading/grain9_6um_indent03_HL_QS_LC")

# Make sure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)
```

#### Run The merrypopins Pipeline

```python
df_pipeline = merrypopins_pipeline(
    text_file,
    save_plot_dir=output_dir,
    trim_margin=30
)
```

#### View Result DataFrame

```python
df_pipeline.head()
```

#### View Result Visualizations

```python
# The pipeline generates plot in the specified output directory for the provided text file.
from PIL import Image
import matplotlib.pyplot as plt

# Load all PNGs from output folder
image_paths = sorted(output_dir.glob("*.png"))

# Only proceed if there are images
if image_paths:
    img = Image.open(image_paths[0])
    plt.figure(figsize=(12, 6))
    plt.imshow(img)
    plt.title(image_paths[0].stem)
    plt.axis('off')
    plt.show()
else:
    print("No plots found in output folder.")
```

---

## Development & Testing

1. Install development requirements:
   ```bash
   pip install -e '.[dev]'
   ```

### 🔧 Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically check code formatting and linting before each commit. This helps ensure consistent code quality across the project.

#### Setup (Run Once)

```bash
# After installing the development dependencies, set up pre-commit hooks:
# This will install the hooks defined in .pre-commit-config.yaml
pre-commit install
```

This sets up a Git hook that will run ruff and black automatically before each commit.

Run Manually

To run all checks on all files:

```bash
pre-commit run --all-files
```

Notes:
- Hooks are defined in .pre-commit-config.yaml.
- You can exclude specific files or directories (e.g., tutorials/) by modifying that config file.

### 🧪 Running Tests
2. Run tests with coverage:
   ```bash
   pytest --cov=merrypopins --cov-report=term-missing
   ```

3. Generate HTML coverage report:
   ```bash
   pytest --cov=merrypopins --cov-report=html
   # open htmlcov/index.html in browser
   ```

---

## Contributing

Contributions are welcome! Please file issues and submit pull requests on [GitHub](https://github.com/SerpRateAI/merrypoppins).

Before submitting a PR:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/foo`).
3. Commit your changes (`git commit -m "feat: add bar"`).
4. Push to the branch (`git push origin feature/foo`).
5. Open a pull request.

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See [LICENSE](LICENSE) for details.

# Indenter

[![Indenter CI Tests](https://github.com/SerpRateAI/indenter/actions/workflows/python-app.yml/badge.svg)](https://github.com/SerpRateAI/indenter/actions/workflows/python-app.yml)
[![PyPI](https://img.shields.io/pypi/v/indenter.svg)](https://pypi.org/project/indenter/)
[![Python](https://img.shields.io/pypi/pyversions/indenter.svg)](https://pypi.org/project/indenter/)
[![codecov](https://codecov.io/gh/SerpRateAI/indenter/branch/main/graph/badge.svg)](https://codecov.io/gh/SerpRateAI/indenter)
[![License](https://img.shields.io/github/license/SerpRateAI/indenter.svg)](LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/indenter.svg)](https://pypi.org/project/indenter/)
[![Issues](https://img.shields.io/github/issues/SerpRateAI/indenter.svg)](https://github.com/SerpRateAI/indenter/issues)
[![Dependencies](https://img.shields.io/librariesio/github/SerpRateAI/indenter)](https://github.com/SerpRateAI/indenter/network/dependencies)
[![Last commit](https://img.shields.io/github/last-commit/SerpRateAI/indenter.svg)](https://github.com/SerpRateAI/indenter/commits/main)
[![Release](https://img.shields.io/github/release-date/SerpRateAI/indenter.svg)](https://github.com/SerpRateAI/indenter/releases)
[![Contributors](https://img.shields.io/github/contributors/SerpRateAI/indenter.svg)](https://github.com/SerpRateAI/indenter/graphs/contributors)

**Indenter** is a library to streamline the workflow of nano‑indentation experiment data processing. It provides five core modules:

- **`load_datasets`**: Load and parse `.txt` measurement files and `.tdm`/`.tdx` metadata files into structured pandas DataFrames. Automatically detects headers, timestamps, and measurement channels.
- **`preprocess`**: Clean and normalize indentation data with filtering, baseline correction, and contact point detection.
- **`locate`**: Identify and extract pop‑in events within indentation curves using event detection algorithms.
- **`statistics`**: Perform statistical analysis and model fitting on located pop‑in events (e.g. frequency, magnitude, distribution).
- **`make_dataset`**: Combine raw measurements, metadata, and analysis outputs into a machine‑learning‑ready dataset.

---

## Installation

```bash
# From PyPI (⚠️ This will not work because package not published yet.)
pip install indenter

# For development
git clone https://github.com/SerpRateAI/indenter.git
cd indenter
pip install -e .
```

Indenter supports Python 3.10+ and depends on:

- `numpy`
- `pandas`
- `scipy`

These are installed automatically via `pip`.

---

## Quickstart

### Importing Indenter Modules

```python
from pathlib import Path
from indenter.load_datasets import load_txt, load_tdm
from indenter.preprocess import default_preprocess, remove_pre_min_load, rescale_data, finalise_contact_index
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

---

## Development & Testing

1. Install development requirements:
   ```bash
   pip install -e '.[dev]'
   ```

2. Run tests with coverage:
   ```bash
   pytest --cov=indenter --cov-report=term-missing
   ```

3. Generate HTML coverage report:
   ```bash
   pytest --cov=indenter --cov-report=html
   # open htmlcov/index.html in browser
   ```

---

## Contributing

Contributions are welcome! Please file issues and submit pull requests on [GitHub](https://github.com/SerpRateAI/indenter).

---

## License

This project is licensed under the **GNU General Public License v3.0**.
See [LICENSE](LICENSE) for details.

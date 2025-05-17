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

**Indenter** is a library to streamline the workflow of nano‑indentation experiment data processing. It provides four core modules:

- **`load_datasets`**: Read and parse `.txt` measurement files and `.tdm`/`.tdx` metadata files into pandas DataFrames, auto‑detecting columns and channels.
- **`preprocess`**: Preprocess raw indentation data into a more usable format, including filtering, smoothing, and normalizing.
- **`locate`**: Identify and extract pop‑in events within indentation curves.
- **`statistics`**: Perform statistical analysis and model fitting on pop‑in data.
- **`make_dataset`**: Enrich raw measurements by combining metadata, located events, and predictive features into cohesive datasets.

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

```python
from pathlib import Path
from indenter.load_datasets import load_txt, load_tdm

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
# The chanel metadata is stored as multiple rows with their respective columns
print(df_tdm_meta_channels.head(50))
```

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

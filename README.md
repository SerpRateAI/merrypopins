# Indenter

**Indenter** is a library to streamline the workflow of nano‑indentation experiment data processing. It provides four core modules:

- **`load_datasets`**: Read and parse `.txt` measurement files and `.tdm`/`.tdx` metadata files into pandas DataFrames, auto‑detecting columns and channels.
- **`transform`**: Transform raw indentation data into a more usable format, including filtering, smoothing, and normalizing.
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

Indenter supports Python 3.9+ and depends on:

- `numpy`
- `pandas`

These are installed automatically via `pip`.

---

## Quickstart

```python
from pathlib import Path
from indenter.load_datasets import load_txt, load_tdm, load_tdx, merge_metadata

# 1) Load indentation data:
data_file = Path("data/experiment1.txt")
df = load_txt(data_file)
print(df.head())
print("Timestamp:", df.attrs['timestamp'])
print("Number of Points:", df.attrs['num_points'])

# 2) Load tdm metadata:
tdm_meta_file = Path("data/experiment1.tdm")
df_tdm_meta = load_tdm(tdm_meta_file)
print(df_tdm_meta[['channel_id','unit','description']])
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

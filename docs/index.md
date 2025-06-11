# Merrypopins

[![merrypopins CI Tests](https://github.com/SerpRateAI/merrypopins/actions/workflows/python-app.yml/badge.svg)](https://github.com/SerpRateAI/merrypopins/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/SerpRateAI/merrypopins/graph/badge.svg)](https://codecov.io/gh/SerpRateAI/merrypopins)
[![📘 Merrypopins Documentation](https://img.shields.io/badge/docs-view-blue?logo=readthedocs)](https://serprateai.github.io/merrypopins/)
[![PyPI](https://img.shields.io/pypi/v/merrypopins.svg)](https://pypi.org/project/merrypopins/)
[![Python](https://img.shields.io/pypi/pyversions/merrypopins.svg)](https://pypi.org/project/merrypopins/)
[![License: GNU](https://img.shields.io/badge/License-GNU-yellow.svg)](LICENSE)
[![Downloads](https://pepy.tech/badge/merrypopins)](https://pepy.tech/project/merrypopins)
[![Issues](https://img.shields.io/github/issues/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/issues)
[![Dependencies](https://img.shields.io/librariesio/github/SerpRateAI/merrypopins)](https://github.com/SerpRateAI/merrypopins/network/dependencies)
[![Dependabot Status](https://img.shields.io/badge/dependabot-enabled-brightgreen.svg)](https://docs.github.com/en/code-security/supply-chain-security/keeping-your-dependencies-updated-automatically)
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

For a quick overview, see the [Quickstart](quickstart.md).

Merrypopins is developed by [Cahit Acar](mailto:c.acar.business@gmail.com), [Anna Marcelissen](mailto:anna.marcelissen@live.nl), [Hugo van Schrojenstein Lantman](mailto:h.w.vanschrojensteinlantman@uu.nl), and [John M. Aiken](mailto:johnm.aiken@gmail.com).
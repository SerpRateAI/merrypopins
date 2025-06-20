# Merrypopins

<p align="center">
  <a href="https://serprateai.github.io/merrypopins/">
    <img src="static/logo-transparent.png" alt="Merrypopins" width="350"/>
  </a>
</p>

[![Merrypopins CI Tests](https://github.com/SerpRateAI/merrypopins/actions/workflows/python-app.yml/badge.svg)](https://github.com/SerpRateAI/merrypopins/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/SerpRateAI/merrypopins/graph/badge.svg)](https://codecov.io/gh/SerpRateAI/merrypopins)
![CodeQL](https://github.com/SerpRateAI/merrypopins/actions/workflows/codeql.yml/badge.svg)
[![📘 Merrypopins Documentation](https://img.shields.io/badge/docs-view-blue?logo=readthedocs)](https://serprateai.github.io/merrypopins/)
[![Merrypopins Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://merrypopins.streamlit.app)
[![PyPI](https://img.shields.io/pypi/v/merrypopins.svg)](https://pypi.org/project/merrypopins/)
[![Python](https://img.shields.io/pypi/pyversions/merrypopins.svg)](https://pypi.org/project/merrypopins/)
[![Docker Pulls](https://img.shields.io/docker/pulls/cacarvuai/merrypopins-app.svg)](https://hub.docker.com/r/cacarvuai/merrypopins-app)
[![Downloads](https://pepy.tech/badge/merrypopins)](https://pepy.tech/project/merrypopins)
[![Issues](https://img.shields.io/github/issues/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/issues)
[![Dependencies](https://img.shields.io/librariesio/github/SerpRateAI/merrypopins)](https://github.com/SerpRateAI/merrypopins/network/dependencies)
[![Dependabot Status](https://img.shields.io/badge/dependabot-enabled-brightgreen.svg)](https://docs.github.com/en/code-security/supply-chain-security/keeping-your-dependencies-updated-automatically)
[![Last commit](https://img.shields.io/github/last-commit/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/commits/main)
[![Release](https://img.shields.io/github/release-date/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/releases)
[![Contributors](https://img.shields.io/github/contributors/SerpRateAI/merrypopins.svg)](https://github.com/SerpRateAI/merrypopins/graphs/contributors)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**merrypopins** is a Python library to streamline the workflow of nano‑indentation experiment data processing, automated pop-in detection and analysis. It provides five core modules:

- **`load_datasets`**: Load and parse `.txt` measurement files and `.tdm`/`.tdx` metadata files into structured pandas DataFrames. Automatically detects headers, timestamps, and measurement channels.
- **`preprocess`**: Clean and normalize indentation data with filtering, baseline correction, and contact point detection.
- **`locate`**: Identify and extract pop‑in events within indentation curves using advanced detection algorithms, including:
  - Isolation Forest anomaly detection
  - CNN Autoencoder reconstruction error
  - Fourier-based derivative outlier detection
  - Savitzky-Golay smoothed gradient thresholds
- **`statistics`**: Perform statistical analysis and model fitting on located pop‑in events (e.g., frequency, magnitude, distribution). The statistics module allows you to compute detailed pop-in statistics, such as:
  - Pop-in statistics (e.g., load-depth and stress-strain metrics)
  - Stress-strain transformation using Kalidindi & Pathak. (2008)
  - Curve-level summary statistics (e.g., total pop-in duration, average time between pop-ins)
  - Pop-in shape statistics like depth jump, average velocity, and curvature
- **`make_dataset`**: Construct enriched datasets by running the full merrypopins pipeline and exporting annotated results and visualizations.

---

## 🌐 Try Merrypopins Library Online

🚀 **Live demo**: explore Merrypopins in your browser! [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://merrypopins.streamlit.app)

The hosted app lets you:

* upload raw `.txt` indentation files (and optional `.tdm/.tdx` metadata),
* tune preprocessing, detection & statistics parameters,
* visualise pop-ins interactively,
* download annotated CSVs + plots.

---

## 🛠 Source Instrumentation
Merrypopins was developed using datasets generated by the Bruker Hysitron TI 990 TriboIndenter — a high-precision nanoindentation platform. The library natively supports .txt and .tdm/.tdx file formats exported by the Hysitron software suite.

<div align="center"> <img src="static/hysitron-ti-990-triboindenter-web-hero-bruker.png" alt="Hysitron TI 990 Nanoindenter" width="300"/> </div>
Typical indentation experiments conducted with the TI 990 include:

- Force-depth curve acquisition at nano/micro scale
- High-resolution pop-in event detection
- Automated test grid data export

The preprocessing and pop-in detection tools in Merrypopins are tuned to handle the structural patterns and noise profiles specific to these datasets.

### Example: Nanoindentation Grain Selection and Deformation

Below are example visualizations from Electron Backscatter Diffraction (EBSD) maps used to select grain areas, followed by indentation marks after testing:

#### ➤ Pre-indentation EBSD with Labeled Grains
<p align="center">
  <img src="static/grain-sample-indentation-areas.png" alt="Grain Selection Map" width="600"/>
</p>

#### ➤ Post-indentation Microstructure with Deformation (Area on Grain 5)
<p align="center">
  <img src="static/grain5-image.png" alt="Grain 5 After Indentation" width="600"/>
</p>

These images highlight the complex deformation behavior analyzed by the `merrypopins` toolset for robust pop-in detection.

---

For a quick overview, see the [Quickstart](quickstart.md).

Merrypopins is developed by [Cahit Acar](mailto:c.acar.business@gmail.com), [Anna Marcelissen](mailto:anna.marcelissen@live.nl), [Hugo van Schrojenstein Lantman](mailto:h.w.vanschrojensteinlantman@uu.nl), and [John M. Aiken](mailto:johnm.aiken@gmail.com).

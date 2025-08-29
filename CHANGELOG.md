# :bookmark_tabs: Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.0] – 2025-06-15 &nbsp;:tada: **“Going Live”**

### Added
- **Core Modules**
  - **`load_datasets`** – Robust parsers for `.txt`, `.tdm`, `.tdx` nano-indentation files.
  - **`preprocess`** – Baseline removal, depth rescaling, contact-point detection, trimming.
  - **`locate`** – Multi-method pop-in detection
    *(Isolation Forest, CNN auto-encoder, Fourier–derivative, Savitzky–Golay)*.
  - **`statistics`** – Comprehensive pop-in analytics:
    load-depth, stress-strain transforms, shape/temporal descriptors, curve-level summaries.
  - **`make_dataset`** – One-shot pipeline that chains **load → preprocess → locate → visualise**.
- **Streamlit App**
  - **`streamlit_app`** – Interactive UI with parameter tuning, visualisations, PNG/CSV export & deployed at <https://merrypopins.streamlit.app>.

- **Infrastructure & Tooling**
  - CI matrix for Python 3.10-3.12, CodeQL, Ruff + Black via pre-commit.
  - 100% test coverage with `pytest`, `coverage`, GitHub Actions badge.
  - Docker image (`cacarvuai/merrypopins-app`) & Compose instructions.
  - Release automation to PyPI & GitHub (version/artefact validation).

### Changed
- **Licence** switched from **GNU GPL-3.0** ➜ **MIT**.
- Branch strategy: feature PRs → `dev`; maintainers merge `dev` → `main` for releases.

## [1.0.1] – 2025-06-20 &nbsp;:bug: **“Patch Release”**
### Fixed
- **Streamlit App**: 
   - Fixed issue with PNG export not working in the latest version of Kaleido, new major version of Kaleido now requires Chromium to be installed. So we have downgraded Kaleido to 0.2.1.

## [1.0.2] – 2025-07-08 &nbsp;:dependabot: **“Dependency Update”**
### Changed
- **Streamlit App**:
    - Updated `kaleido` dependency to latest version.

## [1.0.3] – 2025-07-15 &nbsp;:bug: **“Patch Release”**
### Fixed
- **Streamlit Cloud App**:
    - Fixed issue with PNG export not working in the latest version of Kaleido, new major version of Kaleido now requires Chromium to be installed. So we have downgraded Kaleido to 0.2.1. We have tried to call `kaleido.get_chrome_sync()` to ensure it works in Streamlit Cloud and other environments, but it is not
    possible to call it in the Streamlit Cloud environment, the Streamlit Cloud environment does not allow changing system environment variables like `KALEIDO_CACHE_DIR` or calling `kaleido.get_chrome_sync()`. So we have removed the call to `kaleido.get_chrome_sync()` and set the default format, width, height, and scale for PNG export using `pio.kaleido.scope`. For this we also had to update the `plotly` dependency to `<6.0.0` to ensure compatibility with the `0.2.1` version of `kaleido`.

### Changed
- **Python Compatibility**:
    - As the latest TensorFlow is now compatible with Python 3.13, updated Python compatibility in `pyproject.toml` and classifiers to include Python 3.13.
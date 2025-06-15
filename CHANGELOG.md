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
  - **`streamlit_app`** – Interactive UI with parameter tuning, visualisations, PNG/CSV export & deployed at <https://merrypopins.streamlit.app>.

- **Infrastructure & Tooling**
  - CI matrix for Python 3.10-3.12, CodeQL, Ruff + Black via pre-commit.
  - 100% test coverage with `pytest`, `coverage`, GitHub Actions badge.
  - Docker image (`cacarvuai/merrypopins-app`) & Compose instructions.
  - Release automation to PyPI & GitHub (version/artefact validation).

### Changed
- **Licence** switched from **GNU GPL-3.0** ➜ **MIT**.
- Branch strategy: feature PRs → `dev`; maintainers merge `dev` → `main` for releases.


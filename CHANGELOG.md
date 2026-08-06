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

## [1.0.3] – 2025-07-11 &nbsp;:bug: **“Patch Release”**
### Fixed
- **Streamlit Cloud App**:
    - Fixed issue with PNG export not working in the latest version of Kaleido, new major version of Kaleido now requires Chromium to be installed. So we have downgraded Kaleido to 0.2.1. We have tried to call `kaleido.get_chrome_sync()` to ensure it works in Streamlit Cloud and other environments, but it is not
    possible to call it in the Streamlit Cloud environment, the Streamlit Cloud environment does not allow changing system environment variables like `KALEIDO_CACHE_DIR` or calling `kaleido.get_chrome_sync()`. So we have removed the call to `kaleido.get_chrome_sync()` and set the default format, width, height, and scale for PNG export using `pio.kaleido.scope`. For this we also had to update the `plotly` dependency to `<6.0.0` to ensure compatibility with the `0.2.1` version of `kaleido`.

## [1.0.4] - 2025-08-30 &nbsp;:tada: **“Support for Python 3.13”**
### Changed
- **Python Compatibility**:
    - As the latest TensorFlow is now compatible with Python 3.13, updated Python compatibility in `pyproject.toml` and classifiers to include Python 3.13.
## [1.1.0] - 2026-08-07 &nbsp;:package: **"Optional TensorFlow"**

Revisions made in response to the JOSS peer review
([openjournals/joss-reviews#9933](https://github.com/openjournals/joss-reviews/issues/9933))
and to [#72](https://github.com/SerpRateAI/merrypopins/issues/72).

### Changed
- **TensorFlow is now an optional dependency.** It is only needed by the CNN
  autoencoder detector, one of four methods in `locate`, and it is a
  several-hundred-megabyte install. Install it with `pip install 'merrypopins[cnn]'`.
  The base install is now considerably smaller, and `merrypopins` imports and runs
  without it.
  - `locate.py` imports Keras lazily inside `build_cnn_autoencoder` and
    `detect_popins_cnn`, and raises an `ImportError` naming the install command
    rather than a bare `ModuleNotFoundError`.
  - `default_locate` still enables the CNN by default (`use_cnn=True`), so on a slim
    install it raises rather than quietly returning results from three methods when
    four were requested. Pass `use_cnn=False` for the three-method pipeline.
  - Conda, Docker, dev and Streamlit environments continue to install TensorFlow, so
    the tutorials and the hosted app are unaffected.
- **Type hints** added to the public API of all five modules.
- The four `detect_popins_*` functions now share a single `_apply_trims` helper for
  the edge and post-maximum-load masks they all applied separately. No change in
  behaviour or public API.

### Fixed
- `detect_popins_savgol` and `detect_popins_fd_fourier` no longer flag arbitrary
  points on a perfectly smooth curve. Both score candidates as a departure from the
  mean in units of the derivative's standard deviation. When the curve is smooth the
  derivative is constant, that standard deviation is floating-point noise around zero
  rather than a real scale, and the comparison flagged points essentially at random.
  The shared `_flag_outliers` helper now flags nothing when the spread is negligible
  relative to the signal, which is correct: a perfectly smooth curve has no pop-ins.
  This surfaced as a test failure under newer SciPy and NumPy releases.

### Added
- A pipeline diagram and a detection-method comparison table in the README and docs,
  plus guidance on `popin`, `popin_score` and `popin_confident`.
- Documentation of the optional-extra install path in the README and installation guide.
- A `slim-install` CI job that installs the base package, asserts TensorFlow is absent,
  and runs the tests that do not need it.
- Tests covering the missing-TensorFlow path for `_import_keras`,
  `build_cnn_autoencoder`, `detect_popins_cnn` and `default_locate`.

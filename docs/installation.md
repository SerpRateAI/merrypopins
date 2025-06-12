# Installation

Install from PyPI (once published):

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

All core and development dependencies are tested with Python 3.10 through 3.12.

# Development & Testing

1. Install development requirements:
   ```bash
    # For development (includes dev tools like pytest, black, ruff, etc.)
    pip install -e '.[dev]'
   ```
   This installs the main package and development dependencies listed in pyproject.toml under [project.optional-dependencies].dev

   Optionally, you can install development dependencies via:
   ```bash
   pip install -r requirements-dev.txt
   ```

## 🔧 Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically check code formatting and linting before each commit. This helps ensure consistent code quality across the project.

### Setup (Run Once)

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

## 🧪 Running Tests
2. Run tests with coverage:
   ```bash
   pytest --cov=merrypopins --cov-report=term-missing
   ```
   This command runs all tests in the `tests/` directory and generates a coverage report showing which lines of code were executed during the tests.
   Tests and linting are automatically run on each pull request via GitHub Actions. The CI uses Python 3.10–3.12 and runs pre-commit, pytest, and coverage checks.

3. Generate HTML coverage report:
   ```bash
   pytest --cov=merrypopins --cov-report=html
   # open htmlcov/index.html in browser
   ```
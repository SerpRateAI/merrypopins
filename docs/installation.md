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

We rely on [**pre-commit**](https://pre-commit.com/) to auto-run **ruff** (lint) and **black** (format) against **every** change before it is committed.  
If these checks are **not** executed locally, your PR will fail in CI.

> **🚨 Important:** You must have the `pre-commit` package installed **globally**  
> (`pip install --user pre-commit` or via the project’s *dev* extras) **before** making commits.

### Setup (Run once per clone)

```bash
# 1) Install the tool (only needed if it’s not already on your system)
pip install pre-commit          # or: pip install -e '.[dev]'

# 2) Install the Git hooks defined in .pre-commit-config.yaml
pre-commit install
```

This adds a Git hook that formats / lints the staged files automatically at each `git commit`.

Run Checks Manually

To run all checks on all files:

```bash
pre-commit run --all-files
```

### What if the hook rejects my commit?

If `pre-commit` finds issues (usually formatting via **black** or lint via **ruff**),  
the commit will **abort** and the affected files will be *modified in-place* to satisfy the rules.

1. Open **Source Control** (e.g. the Git sidebar in VS Code).  
2. You will see the *updated* (but **unstaged**) files.
3. Click the **➕** (stage) button next to each fixed file *or* `git add <file>`.
4. Re-run `git commit` – it should now succeed.
5. Finally, push your branch to the remote.

> Tip: always run `pre-commit run --all-files` before making a commit to catch issues early.

Notes:
- Hooks are defined in `.pre-commit-config.yaml`.
- You can exclude specific files or directories (e.g., `tutorials/`) by modifying the config file `.pre-commit-config.yaml`.
- CI will re-run the same hooks; commits that bypass them locally will be rejected.

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
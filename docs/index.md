# Indenter

> Streamlining nano‑indentation data processing workflows.

[![Indenter CI Tests](https://github.com/SerpRateAI/indenter/actions/workflows/python-app.yml/badge.svg)](https://github.com/SerpRateAI/indenter/actions/workflows/python-app.yml)
[![PyPI](https://img.shields.io/pypi/v/indenter.svg)](https://pypi.org/project/indenter/)
[![Python](https://img.shields.io/pypi/pyversions/indenter.svg)](https://pypi.org/project/indenter/)
[![codecov](https://codecov.io/gh/SerpRateAI/indenter/branch/main/graph/badge.svg)](https://codecov.io/gh/SerpRateAI/indenter)
[![License](https://img.shields.io/github/license/SerpRateAI/indenter.svg)](../LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/indenter.svg)](https://pypi.org/project/indenter/)
[![Issues](https://img.shields.io/github/issues/SerpRateAI/indenter.svg)](https://github.com/SerpRateAI/indenter/issues)
[![Dependencies](https://img.shields.io/librariesio/github/SerpRateAI/indenter)](https://github.com/SerpRateAI/indenter/network/dependencies)
[![Last commit](https://img.shields.io/github/last-commit/SerpRateAI/indenter.svg)](https://github.com/SerpRateAI/indenter/commits/main)
[![Release](https://img.shields.io/github/release-date/SerpRateAI/indenter.svg)](https://github.com/SerpRateAI/indenter/releases)
[![Contributors](https://img.shields.io/github/contributors/SerpRateAI/indenter.svg)](https://github.com/SerpRateAI/indenter/graphs/contributors)

**Indenter** is a library to streamline the workflow of nano‑indentation experiment data processing. It provides five core modules:

- **`load_datasets`**: Read and parse measurement files into pandas DataFrames.
- **`preprocess`**: Clean, filter, and normalize raw indentation data.
- **`locate`**: Identify and extract pop‑in events.
- **`statistics`**: Perform statistical analysis and model fitting.
- **`make_dataset`**: Combine metadata and features into cohesive datasets.

For a quick overview, see the [Quickstart](quickstart.md).
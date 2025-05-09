# Installation

Install from PyPI (once published):

```bash
pip install indenter
```
Or for development from source:

```bash
git clone https://github.com/SerpRateAI/indenter.git
cd indenter
pip install -e .
```
Indenter supports Python 3.10+ and depends on:
- `numpy`
- `pandas`
These are installed automatically via `pip`.

# Testing

1. Install development requirements:
```bash
   pip install -e '.[dev]'
```

2. Run tests:
```bash
   pytest
```

3. Run tests with coverage:
```bash
    pytest --cov=indenter --cov-report=term-missing
```

4. Run tests with coverage and generate HTML report:
```bash
    pytest --cov=indenter --cov-report=html
```
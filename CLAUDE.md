# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
uv sync              # Install all dependencies including dev deps (creates .venv if it doesn't exist)
source .venv/bin/activate  # Activate virtual environment (Linux/Mac)
pre-commit install  # Install pre-commit hooks
```

**Important Notes:**
- If `.venv` directory doesn't exist, `uv sync` will create it automatically
- `uv sync` installs both main and dev dependencies by default
- Always activate the virtual environment before running git commands to ensure pre-commit hooks work properly
- When creating PRs, remember to update the version in `pyproject.toml`

### Testing
```bash
uv run pytest                    # Run all tests
uv run pytest tests/test_*.py    # Run specific test file
uv run pytest -k "test_name"    # Run tests matching pattern
uv run coverage run -m pytest && uv run coverage report  # Run tests with coverage
```

### Code Quality
```bash
uv run ruff check               # Lint code
uv run ruff check --fix         # Auto-fix linting issues
uv run mypy bio_curve_fit/      # Type checking
uv run isort bio_curve_fit/     # Sort imports
```

## Architecture

This is a Python package that provides curve fitting algorithms for biological assays, following the scikit-learn API pattern.

### Core Structure
- `bio_curve_fit/base.py`: Abstract base class `BaseStandardCurve` defining the interface for all curve models
- `bio_curve_fit/logistic.py`: Contains `LogisticRegression` abstract base class and concrete implementations (`FourParamLogistic`, `FiveParamLogistic`)
- `bio_curve_fit/plotting.py`: Plotting utilities for visualizing curves and limits of detection

### Key Concepts
- All models inherit from `BaseStandardCurve` and implement `predict()` and `predict_inverse()` methods
- Models follow scikit-learn conventions with `fit()`, `predict()`, and parameter attributes ending in `_`
- Limits of Detection (LOD) are calculated automatically: `LLOD`/`ULOD` for concentration, `LLOD_y_`/`ULOD_y_` for response
- Uses scipy.optimize.curve_fit for non-linear regression with optional inverse variance weighting

### Model Usage Pattern
```python
model = FourParamLogistic()
model.fit(concentrations, responses)
predictions = model.predict(new_concentrations)        # Forward prediction
concentrations = model.predict_inverse(new_responses)  # Inverse prediction
```

The package is designed for biologists with minimal programming experience, so examples should be simple and well-documented.
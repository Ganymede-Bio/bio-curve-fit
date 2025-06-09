# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup
```bash
poetry install       # Install dependencies
poetry shell        # Activate virtual environment
pre-commit install  # Install pre-commit hooks
```

### Testing
```bash
poetry run pytest                    # Run all tests
poetry run pytest tests/test_*.py    # Run specific test file
poetry run pytest -k "test_name"    # Run tests matching pattern
coverage run -m pytest && coverage report  # Run tests with coverage
```

### Code Quality
```bash
poetry run ruff check               # Lint code
poetry run ruff check --fix         # Auto-fix linting issues
poetry run mypy bio_curve_fit/      # Type checking
poetry run isort bio_curve_fit/     # Sort imports
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
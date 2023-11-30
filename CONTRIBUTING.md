## Development Setup

This project uses [Poetry](https://python-poetry.org/docs/) for dependency management.

To install dependencies, run the following commands from the root directory of the project:

```shell
poetry install
poetry shell
```

Install pre-commit hooks for automatic style and type checking:

```console
$ pre-commit install
```

## Tests

Run `pytest` from the root directory of the project. Please add tests for any new functionality and keep coverage above ~95%. To check test coverage, run:

```shell
coverage run -m pytest
coverage report
```

## Style

This project uses black, ruff, isort, and mypy. Try to get typing working as best you can, but feel free to add `#type ignore` comments if it feels like more work is going toward satisfying mypy than making this package useful.

## TODOs

- [ ] Implement other [common models](https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_models_built-in_to_prism.htm)
	- [ ] 5PL logistic model is a top priority.
- [ ] Ensure compatibility with other versions of python ex 3.8, and other operating systems.
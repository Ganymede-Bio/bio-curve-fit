
# bio-curve-fit

A Python package for fitting dose-response and standard curves. Designed to follow the api of [scikit-learn](https://scikit-learn.org/stable/).

## Installation

```shell
pip install bio-curve-fit
```

Example usage:

```python
from bio_curve_fit.logistic import FourPLLogistic

# Instantiate model
model = FourPLLogistic()

# create some example data
standard_concentrations = [0, 1, 2, 3, 4, 5]
standard_responses = [0, 0.5, 0.75, 0.9, 0.95, 1]

# fit the model using an optional inverse variance weight function (1/y^2)
model = FourPLLogistic().fit(
	standard_concentrations, 
	standard_responses, 
	weight_func=FourPLLogistic.inverse_variance_weight_function
)

# interpolate the response at given concentrations
model.predict([1.5, 2.5])

# interpolate the concentration at given responses
model.predict_inverse([0.1, 1.0])

```

## Contributing

Contributions are welcome! We built this package to be useful for our own work, but we know there is more to add.
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

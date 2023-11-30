
# bio-curve-fit

A Python package for fitting common dose-response and standard curve models. Designed to follow the [scikit-learn](https://scikit-learn.org/stable/) api.

## Quickstart 

### Installation

```shell
pip install bio-curve-fit
```

### Example usage:

```python
from bio_curve_fit.logistic import FourPLLogistic

# Instantiate model
model = FourPLLogistic()

# create some example data
standard_concentrations = [1, 2, 3, 4, 5]
standard_responses = [0.5, 0.55, 0.9, 1.25, 1.55]


# fit the model using an optional inverse variance weight function (1/y^2)
model = FourPLLogistic().fit(
	standard_concentrations, 
	standard_responses, 
)

# interpolate the response at given concentrations
model.predict([1.5, 2.5])

# interpolate the concentration at given responses
model.predict_inverse([0.1, 1.0])

```

Calculate and plot the curve and limits of detection:

```python
plot_standard_curve(standard_concentrations, standard_responses, model, show_plot=True)
```

![standard curve](./examples/readme_fit.png)

## Examples

See [examples](./examples) for more detailed usage.

## Contributing

Contributions are welcome! We built this package to be useful for our own work, but we know there is more to add.
Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

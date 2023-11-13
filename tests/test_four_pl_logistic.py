import numpy as np
import matplotlib.pyplot as plt
from bio_curve_fit.four_pl_logistic import FourPLLogistic

TEST_PARAMS = [0.0, 1.0, 2.0, 3.0]

x_data = np.linspace(0, 10, 100)
y_data = FourPLLogistic.four_param_logistic(
    x_data + np.random.normal(0.0, 0.1 * x_data, len(x_data)), *TEST_PARAMS
)

model = FourPLLogistic().fit(
    x_data, y_data, weight_func=FourPLLogistic.inverse_variance_weight_function
)

# Extract the fitted parameters
params = model.get_params()

assert np.isclose(params, TEST_PARAMS, rtol=.01).all()

# Generate y-data based on the fitted parameters
y_fitted = model.predict(x_data)

# Plot the data and the fitted curve
plt.scatter(x_data, y_data, label="Data")
plt.plot(x_data, y_fitted, label="Fitted curve", color="red")
plt.legend()
plt.xlabel("x")
plt.ylabel("Response")
plt.title("4PL Curve Fit")
plt.show()

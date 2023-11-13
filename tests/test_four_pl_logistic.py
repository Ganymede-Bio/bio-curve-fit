import numpy as np
import matplotlib.pyplot as plt
from bio_curve_fit.four_pl_logistic import FourPLLogistic
import io
from matplotlib.ticker import ScalarFormatter

# set a seed for reproducibility
np.random.seed(42)

def test_fit():
    TEST_PARAMS = [1.0, 1.0, 2.0, 3.0]

    x_data = np.logspace(0.00001, 7, 100, base=np.e)
    # generate y-data based on the test parameters
    y_data = FourPLLogistic.four_param_logistic(
        x_data + np.random.normal(0.0, 0.1 * x_data, len(x_data)), *TEST_PARAMS
    )

    model = FourPLLogistic().fit(
        x_data, y_data, weight_func=FourPLLogistic.inverse_variance_weight_function
    )

    # Extract the fitted parameters
    params = model.get_params()

    assert np.isclose(params, TEST_PARAMS, rtol=0.4).all()  # type: ignore
    r2 = model.score(x_data, y_data)
    assert r2 > 0.95
    plot_curve(x_data, y_data, model)


def plot_curve(x_data, y_data, fitted_model: FourPLLogistic) -> bytes:
    # Generate y-data based on the fitted parameters
    # Plot the data and the fitted curve
    # set x-axis to log scale
    epsilon = 0.1
    x_min = np.log10(max(min(x_data), epsilon))
    x = np.logspace(x_min, np.log10(max(x_data)), 100)
    y_pred = fitted_model.predict(x)
    # set scales to log
    plt.xscale("log")
    plt.yscale("log")

    plt.scatter(x_data, y_data, label="Data")
    plt.plot(x, y_pred, label="Fitted curve", color="red")
    plt.legend()
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel("concentration")
    plt.ylabel("Response")
    plt.title("4PL Curve Fit")

    # set horizontal and vertical lines for ULOD and LLOD
    llod, ulod = fitted_model.calculate_lod(x_data, y_data)
    plt.axhline(llod, color="red", linestyle="--")
    plt.axhline(ulod, color="red", linestyle="--")
    print("LLOD", llod)
    print("ULOD", ulod)

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.show()
    plt.clf()
    buf.seek(0)
    return buf.read()

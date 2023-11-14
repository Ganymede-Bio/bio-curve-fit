import numpy as np
import pandas as pd
from bio_curve_fit.four_pl_logistic import FourPLLogistic
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import io


def plot_curve(x_data, y_data, fitted_model: FourPLLogistic) -> bytes:
    """
    Generate a plot of the data and the fitted curve.
    """
    # Plot the data and the fitted curve
    # set x-axis to log scale
    # set scales to log
    plt.xscale("log")
    plt.yscale("log")
    data = pd.DataFrame({"x": x_data, "y": y_data})
    # remove zeros from x_data
    filtered_data = data[data["x"] > 0]

    # Plot the fitted curve
    epsilon = 0.01
    x_min = np.log10(max(min(x_data), epsilon))
    x_max = max(x_data) * 2
    x = np.logspace(x_min, np.log10(x_max), 100)  # type: ignore
    # Generate y-data based on the fitted parameters
    y_pred = fitted_model.predict(x)

    plt.plot(x, y_pred, label="Fitted curve", color="red")
    plt.scatter(filtered_data["x"], filtered_data["y"], label="Data", s=12)
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel("concentration")
    plt.ylabel("Response")
    plt.title("4PL Curve Fit")

    # set horizontal and vertical lines for ULOD and LLOD
    _, _, llod_x, ulod_x = fitted_model.calculate_lod(x_data, y_data)
    plt.axhline(llod_x, color="red", linestyle="--", label="LLOD")
    plt.axhline(ulod_x, color="blue", linestyle="--", label="ULOD")
    plt.legend()

    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.show()
    plt.clf()
    buf.seek(0)
    return buf.read()

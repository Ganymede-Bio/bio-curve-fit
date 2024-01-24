import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from .base import BaseStandardCurve


def plot_standard_curve(
    x_data,
    y_data,
    fitted_model: BaseStandardCurve,
    title="Standard Curve Fit",
    x_label="Concentration",
    y_label="Response",
    show_plot: bool = False,
) -> bytes:
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
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # set horizontal and vertical lines for ULOD and LLOD
    llod_response, ulod_response = fitted_model.LLOD_y_, fitted_model.ULOD_y_
    plt.axhline(llod_response, color="red", linestyle="--", label="LLOD")  # type: ignore
    plt.axhline(ulod_response, color="blue", linestyle="--", label="ULOD")  # type: ignore
    plt.legend()
    plt.tight_layout()
    if show_plot:
        plt.show()
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.clf()
    buf.seek(0)
    return buf.read()

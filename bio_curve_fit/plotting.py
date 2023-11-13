import numpy as np
from bio_curve_fit.four_pl_logistic import FourPLLogistic
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import io

def plot_curve(x_data, y_data, fitted_model: FourPLLogistic) -> bytes:
    """
    Generate a plot of the data and the fitted curve.
    """
    # Generate y-data based on the fitted parameters
    # Plot the data and the fitted curve
    # set x-axis to log scale
    epsilon = 0.1
    x_min = np.log10(max(min(x_data), epsilon))
    x = np.logspace(x_min, np.log10(max(x_data)), 100)  # type: ignore 
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
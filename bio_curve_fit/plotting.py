import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatter

from .base import BaseStandardCurve


def plot_standard_curve(
    x_data,
    y_data,
    fitted_model: BaseStandardCurve,
    title="Standard Curve Fit",
    x_label="Concentration",
    y_label="Response",
    show_plot: bool = False,
    curve_kwargs=None,  # kwargs for the fitted curve plot
    data_kwargs=None,  # kwargs for the data scatter plot
    llod_kwargs=None,  # kwargs for the LLOD line
    ulod_kwargs=None,  # kwargs for the ULOD line
    **plot_kwargs  # kwargs for general plot adjustments
) -> bytes:
    """
    Generate a plot of the data and the fitted curve with customizable plotting parameters.

    Parameters:
    - x_data (iterable): X-axis data points.
    - y_data (iterable): Y-axis data points corresponding to x_data.
    - fitted_model (BaseStandardCurve): A fitted model instance that provides prediction and LLOD/ULOD values.
    - title (str, optional): Title of the plot. Default is "Standard Curve Fit".
    - x_label (str, optional): Label for the X-axis. Default is "Concentration".
    - y_label (str, optional): Label for the Y-axis. Default is "Response".
    - show_plot (bool, optional): If True, display the plot. Default is False.
    - curve_kwargs (dict, optional): Keyword arguments for the plot function for the fitted curve. Default is {'label': 'Fitted curve', 'color': 'red'}.
    - data_kwargs (dict, optional): Keyword arguments for the scatter function for data points. Default is {'label': 'Data', 's': 12}.
    - llod_kwargs (dict, optional): Keyword arguments for the axhline function for the Lower Limit of Detection line. Default is {'color': 'red', 'linestyle': '--', 'label': 'LLOD'}.
    - ulod_kwargs (dict, optional): Keyword arguments for the axhline function for the Upper Limit of Detection line. Default is {'color': 'blue', 'linestyle': '--', 'label': 'ULOD'}.
    - plot_kwargs (dict, optional): General keyword arguments for further plot customizations. This can include 'title_kwargs' for title properties and 'savefig_kwargs' for savefig properties, 'formatter' for the x-axis formatter (Default is Log), and 'xscale' and 'yscale' for the plot scale (Default is 'log').

    Returns:
    - bytes: A bytes object containing the plot image in PNG format.

    Example Usage:
    plot_standard_curve(x_data, y_data, fitted_model, show_plot=True, curve_kwargs={'color': 'green', 'linestyle': '--'}, data_kwargs={'color': 'blue', 'marker': 'o'}, llod_kwargs={'color': 'orange'}, ulod_kwargs={'color': 'purple'})

    This function allows extensive customization of the plot's appearance by adjusting properties of the curve, data points, LLOD line, ULOD line, and overall plot aesthetics through various keyword arguments.
    """
    # Default keyword argument dictionaries
    if curve_kwargs is None:
        curve_kwargs = {"label": "Fitted curve", "color": "red"}
    if data_kwargs is None:
        data_kwargs = {"label": "Data", "s": 12}
    if llod_kwargs is None:
        llod_kwargs = {"color": "red", "linestyle": "--", "label": "LLOD"}
    if ulod_kwargs is None:
        ulod_kwargs = {"color": "blue", "linestyle": "--", "label": "ULOD"}

    plt.xscale(plot_kwargs.get("xscale", "log"))
    plt.yscale(plot_kwargs.get("yscale", "log"))
    data = pd.DataFrame({"x": x_data, "y": y_data})
    filtered_data = data[data["x"] > 0]

    epsilon = 0.01
    x_min = np.log10(max(min(x_data), epsilon))
    x_max = max(x_data) * 2
    x = np.logspace(x_min, np.log10(x_max), 100)  # type: ignore
    y_pred = fitted_model.predict(x)

    plt.plot(x, y_pred, **curve_kwargs)
    plt.scatter(filtered_data["x"], filtered_data["y"], **data_kwargs)

    formatter = plot_kwargs.get("formatter", LogFormatter())
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, **plot_kwargs.get("title_kwargs", {}))

    llod_response, ulod_response = fitted_model.LLOD_y_, fitted_model.ULOD_y_
    if llod_response is not None:
        plt.axhline(llod_response, **llod_kwargs)  # type: ignore
    if ulod_response is not None:
        plt.axhline(ulod_response, **ulod_kwargs)  # type: ignore

    plt.legend()
    plt.tight_layout()

    if show_plot:
        plt.show()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", **plot_kwargs.get("savefig_kwargs", {}))
    plt.clf()
    buf.seek(0)
    return buf.read()

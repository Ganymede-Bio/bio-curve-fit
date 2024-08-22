import io
from typing import Tuple

import matplotlib
import matplotlib.figure
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
    - y_data (iterable | None): Y-axis data points corresponding to x_data. Can set this to None if you only want to plot the fitted curve for a given range of x-values specified in `x_data`.
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
    - fig, ax: The matplotlib figure and axes objects.

    Example Usage:
    fig, ax = plot_standard_curve(x_data, y_data, fitted_model, show_plot=True, curve_kwargs={'color': 'green', 'linestyle': '--'}, data_kwargs={'color': 'blue', 'marker': 'o'}, llod_kwargs={'color': 'orange'}, ulod_kwargs={'color': 'purple'})
    ax.set_xlim([0, 10])  # Example of additional modification

    This function allows extensive customization of the plot's appearance by adjusting properties of the curve, data points, LLOD line, ULOD line, and overall plot aesthetics through various keyword arguments. For more advanced customization, consider using plot_standard_curve_figure instead, which returns the figure and axes objects for further modification.
    """

    fig, ax = plot_standard_curve_figure(
        x_data,
        y_data,
        fitted_model,
        title=title,
        x_label=x_label,
        y_label=y_label,
        curve_kwargs=curve_kwargs,
        data_kwargs=data_kwargs,
        llod_kwargs=llod_kwargs,
        ulod_kwargs=ulod_kwargs,
        **plot_kwargs
    )
    if show_plot:
        plt.show()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", **plot_kwargs.get("savefig_kwargs", {}))
    plt.clf()
    buf.seek(0)
    return buf.read()


def plot_standard_curve_figure(
    x_data,
    y_data,
    fitted_model: BaseStandardCurve,
    title="Standard Curve Fit",
    x_label="Concentration",
    y_label="Response",
    curve_kwargs=None,  # kwargs for the fitted curve plot
    data_kwargs=None,  # kwargs for the data scatter plot
    llod_kwargs=None,  # kwargs for the LLOD line
    ulod_kwargs=None,  # kwargs for the ULOD line
    **plot_kwargs  # kwargs for general plot adjustments
) -> Tuple[matplotlib.figure.Figure, plt.Axes]:
    """
    Same as plot_standard_curve, but returns the figure and axes objects instead of saving the plot to a file
    for easier customization and further modification.

    Example Usage:

    ```
    # Add text labels to the data points
    from adjustText import adjust_text

    fig, ax = plot_standard_curve_figure(x_data, y_data, fitted_model)
    texts = []
    for x, y in zip(x_data, y_data):
        texts.append(ax.text(x, y, f"x={x:.2f}, y={y:.2f})", fontsize=13, ha="right"))

    # Adjust text labels to avoid overlap
    adjust_text(texts, ax=ax)
    ```
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

    fig, ax = plt.subplots()

    ax.set_xscale(plot_kwargs.get("xscale", "log"))
    ax.set_yscale(plot_kwargs.get("yscale", "log"))

    EPSILON = 0.01
    x_min = np.log10(max(min(x_data), EPSILON))
    x_max = max(x_data) * 2
    x = np.logspace(x_min, np.log10(x_max), 100)  # type: ignore
    y_pred = fitted_model.predict(x)

    ax.plot(x, y_pred, **curve_kwargs)
    if y_data is not None:
        data = pd.DataFrame({"x": x_data, "y": y_data})
        filtered_data = data[data["x"] > 0]
        ax.scatter(filtered_data["x"], filtered_data["y"], **data_kwargs)

    formatter = plot_kwargs.get("formatter", LogFormatter())
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, **plot_kwargs.get("title_kwargs", {}))

    llod_response, ulod_response = fitted_model.LLOD_y_, fitted_model.ULOD_y_
    if llod_response is not None:
        ax.axhline(llod_response, **llod_kwargs)  # type: ignore
    if ulod_response is not None:
        ax.axhline(ulod_response, **ulod_kwargs)  # type: ignore

    ax.legend()
    fig.tight_layout()

    return fig, ax

import io
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from adjustText import adjust_text
from matplotlib.testing.compare import compare_images

from bio_curve_fit.logistic import FourPLLogistic
from bio_curve_fit.plotting import plot_standard_curve, plot_standard_curve_figure

# set a seed for reproducibility
np.random.seed(42)


def compare_bytes_to_reference(img_bytes, relative_reference_path):
    """
    Helper function to compare an image in bytes format to a reference image
    """
    current_dir = os.path.dirname(__file__)
    full_path = os.path.join(current_dir, relative_reference_path)
    # create tmp file to save the image
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        image_path = f.name
        f.write(img_bytes)
        # compare the image to the reference image
        # TODO: this is only passing at very high tol when running in CI
        # but was passing with a much lower tol when running locally
        # For now dynamically set the tolerance based on the environment
        tolerance = int(os.getenv("PLOT_COMPARISON_TOLERANCE", 0.00001))
        comparison_result = compare_images(full_path, image_path, tol=tolerance)
    if comparison_result is not None:
        raise AssertionError(comparison_result)


def test_fit_and_plot():
    TEST_PARAMS = [1.0, 1.0, 2.0, 3.0]

    x_data = np.logspace(0.00001, 7, 100, base=np.e)  # type: ignore
    # generate y-data based on the test parameters
    y_data = FourPLLogistic._logistic_model(
        x_data + np.random.normal(0.0, 0.1 * x_data, len(x_data)), *TEST_PARAMS
    )

    model = FourPLLogistic().fit(
        x_data, y_data, weight_func=FourPLLogistic.inverse_variance_weight_function
    )

    # model should recover parameters used to generate the data
    params = list(model.get_params().values())
    assert np.isclose(params, TEST_PARAMS, rtol=0.4).all()  # type: ignore

    r2 = model.score(x_data, y_data)
    assert r2 > 0.995

    # test plotting
    img_bytes = plot_standard_curve(
        x_data,
        y_data,
        model,
        llod_kwargs={"color": "pink"},
    )
    # create tmp file to save the image
    compare_bytes_to_reference(img_bytes, "reference_plots/test_fit_and_plot.png")

    # test __repr__
    print(model)


test_y = pd.Series(
    [
        223747,
        214105,
        61193,
        61831,
        16290,
        14151,
        4097,
        3895,
        1587,
        1485,
        896,
        771,
        743,
        674,
        653,
        562,
        634,
        642,
    ]
)

test_x = pd.Series(
    [
        1360,
        1360,
        340,
        340,
        85,
        85,
        21.25,
        21.25,
        5.3125,
        5.3125,
        1.328125,
        1.328125,
        0.33203125,
        0.33203125,
        0.0830078125,
        0.0830078125,
        0.0,
        0.0,
    ]
)


def test_fit2():
    model = FourPLLogistic().fit(
        test_x,
        test_y,
        weight_func=FourPLLogistic.inverse_variance_weight_function,
    )
    print("Params:", model.get_params())
    print(model.predict_inverse(0.1))
    plot_standard_curve(test_x, test_y, model)
    assert model.score(test_x, test_y) > 0.995  # type: ignore
    print(model.ULOD_y_, model.LLOD_y_)

    assert np.isclose(model.ULOD_y_, 220006.8397685415, rtol=0.1)  # type: ignore
    assert np.isclose(model.LLOD_y_, 798.7000577483678, rtol=0.1)  # type: ignore


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_readme_example():
    """
    ensure that the example in the README works
    """
    # Instantiate model
    model = FourPLLogistic()

    # create some example data
    standard_concentrations = [1, 2, 3, 4, 5]
    standard_responses = [0.5, 0.55, 0.9, 1.25, 1.55]

    # fit the model
    model.fit(
        standard_concentrations,
        standard_responses,
    )

    # interpolate the response at given concentrations
    values = model.predict([1.5, 2.5])
    assert pd.notna(values).all()

    # interpolate the concentration at given responses
    values = model.predict_inverse([0.1, 1.0])
    assert pd.notna(values).all()  # type: ignore
    img_bytes = plot_standard_curve(
        standard_concentrations, standard_responses, model, title="4PL Curve Fit"
    )

    compare_bytes_to_reference(img_bytes, "../examples/readme_fit.png")
    plt.clf()

    fig, ax = plot_standard_curve_figure(
        standard_concentrations, standard_responses, model
    )
    texts = []
    for x, y in zip(standard_concentrations, standard_responses):
        texts.append(ax.text(x, y, f"x={x:.2f}, y={y:.2f}", fontsize=13, ha="right"))
    # Adjust text labels to avoid overlap
    adjust_text(texts, ax=ax)

    # save the figure
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.clf()
    buf.seek(0)
    img2_bytes = buf.read()

    compare_bytes_to_reference(img2_bytes, "../examples/readme_fit_labels.png")


def test_limits():
    param_dict = {"A": 2, "B": 1.3, "C": 1, "D": 400}
    model = FourPLLogistic(params=param_dict)
    y = [1.5, 4, 401]
    x = model.predict_inverse(y)
    assert x[0] == 0  # type: ignore
    assert x[1] > 0  # type: ignore
    assert np.isnan(x[2])  # type: ignore
    x2 = model.predict_inverse(y, enforce_limits=False)
    # should not enforce limits
    assert x2[0] < 0  # type: ignore
    assert x2[1] > 0  # type: ignore
    assert x2[2] < 0  # type: ignore


def test_std_dev():
    model = FourPLLogistic().fit(
        test_x,
        test_y,
    )
    ci = model.predict_confidence_band(test_x)
    pi = model.predict_prediction_band(test_x, test_y)

    print(ci)
    print(pi)

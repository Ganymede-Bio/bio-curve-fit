import io
import os
import tempfile
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from adjustText import adjust_text  # type: ignore
from matplotlib.testing.compare import compare_images

from bio_curve_fit.logistic import FourParamLogistic
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
        tolerance = float(os.getenv("PLOT_COMPARISON_TOLERANCE", "25"))
        comparison_result = compare_images(full_path, image_path, tol=tolerance)
    if comparison_result is not None:
        raise AssertionError(comparison_result)


def test_fit_and_plot():
    np.random.seed(42)  # reset in test func as it seems to gett reset in CI
    TEST_PARAMS = {"A": 1, "B": 1, "C": 2, "D": 3}

    x_data = np.logspace(-6, 7, 100, base=np.e)  # type: ignore
    # generate y-data based on the test parameters
    y_data = FourParamLogistic._logistic_model(x_data, **TEST_PARAMS)
    y_data_with_noise = y_data + np.random.normal(0, 0.01, len(y_data))

    model = FourParamLogistic()
    with pytest.raises(ValueError):
        model.predict(x_data)

    model.fit(
        x_data,
        y_data_with_noise,
        weight_func=FourParamLogistic.inverse_variance_weight_function,
    )

    fitted_params = model.get_params()
    # model should recover parameters used to generate the data
    for key, value in TEST_PARAMS.items():
        assert np.isclose(fitted_params[key], value, rtol=0.08)  # type: ignore

    r2 = model.score(x_data, y_data)
    assert r2 > 0.995

    # test plotting
    img_bytes = plot_standard_curve(
        x_data,
        y_data_with_noise,
        model,
        llod_kwargs={"color": "pink"},
        xscale="symlog",
    )
    # create tmp file to save the image
    compare_bytes_to_reference(img_bytes, "reference_plots/test_fit_and_plot.png")

    # test __repr__
    print(model)
    print(model.get_params())


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
    model = FourParamLogistic().fit(
        test_x,
        test_y,
        weight_func=FourParamLogistic.inverse_variance_weight_function,
    )
    print("Params:", model.get_params())
    print(model.predict_inverse(0.1))
    plot_standard_curve(test_x, test_y, model)
    assert model.score(test_x, test_y) > 0.995  # type: ignore
    print(model.ULOD_y_, model.LLOD_y_)

    assert np.isclose(model.ULOD_y_, 220006.8397685415, rtol=0.1)  # type: ignore
    assert np.isclose(model.LLOD_y_, 798.7000577483678, rtol=0.1)  # type: ignore


def test_parameter_constraints():
    """Test that parameter constraints work correctly."""
    np.random.seed(42)
    TEST_PARAMS = {"A": 1, "B": 1, "C": 2, "D": 3}

    x_data = np.logspace(-6, 7, 100, base=np.e)
    y_data = FourParamLogistic._logistic_model(x_data, **TEST_PARAMS)
    y_data_with_noise = y_data + np.random.normal(0, 0.01, len(y_data))

    # Test fixing B parameter (3PL regression)
    model_3pl = FourParamLogistic(B=1.0)
    model_3pl.fit(x_data, y_data_with_noise)

    # B should remain fixed at 1.0
    assert model_3pl.B == 1.0

    # Other parameters should be fitted
    assert model_3pl.A is not None and model_3pl.A != 1.0  # unlikely to be exactly 1.0
    assert model_3pl.C is not None and model_3pl.C != 2.0  # unlikely to be exactly 2.0
    assert model_3pl.D is not None and model_3pl.D != 3.0  # unlikely to be exactly 3.0

    # Model should still fit reasonably well
    r2 = model_3pl.score(x_data, y_data)
    assert r2 > 0.99

    # Test fixing multiple parameters
    model_fixed_two = FourParamLogistic(A=1.0, D=3.0)
    model_fixed_two.fit(x_data, y_data_with_noise)

    # Fixed parameters should remain unchanged
    assert model_fixed_two.A == 1.0
    assert model_fixed_two.D == 3.0

    # Free parameters should be fitted
    assert model_fixed_two.B is not None
    assert model_fixed_two.C is not None

    # Test fixing all parameters (no fitting needed)
    model_all_fixed = FourParamLogistic(A=1.0, B=1.0, C=2.0, D=3.0)
    model_all_fixed.fit(x_data, y_data_with_noise)

    # All parameters should remain as set
    assert model_all_fixed.A == 1.0
    assert model_all_fixed.B == 1.0
    assert model_all_fixed.C == 2.0
    assert model_all_fixed.D == 3.0

    # Should be able to predict
    predictions = model_all_fixed.predict(x_data[:5])
    assert len(predictions) == 5


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_readme_example():
    """
    ensure that the example in the README works
    """
    # Instantiate model
    model = FourParamLogistic()

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
        standard_concentrations,
        standard_responses,
        model,
        title="4PL Curve Fit",
        # show_plot=True,
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
    model = FourParamLogistic(**param_dict)
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
    # TODO: this test is not very useful
    model = FourParamLogistic().fit(
        test_x,
        test_y,
    )
    ci = model.predict_confidence_band(test_x)
    pi = model.predict_prediction_band(test_x, test_y)

    print(ci)
    print(pi)

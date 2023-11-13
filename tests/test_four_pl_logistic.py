import numpy as np
from bio_curve_fit.four_pl_logistic import FourPLLogistic
from bio_curve_fit.plotting import plot_curve

# set a seed for reproducibility
np.random.seed(42)


def test_fit():
    TEST_PARAMS = [1.0, 1.0, 2.0, 3.0]

    x_data = np.logspace(0.00001, 7, 100, base=np.e)  # type: ignore
    # generate y-data based on the test parameters
    y_data = FourPLLogistic.four_param_logistic(
        x_data + np.random.normal(0.0, 0.1 * x_data, len(x_data)), *TEST_PARAMS
    )

    model = FourPLLogistic().fit(
        x_data, y_data, weight_func=FourPLLogistic.inverse_variance_weight_function
    )

    params = model.get_params()
    assert np.isclose(params, TEST_PARAMS, rtol=0.4).all()  # type: ignore

    r2 = model.score(x_data, y_data)
    assert r2 > 0.95

    plot_curve(x_data, y_data, model)

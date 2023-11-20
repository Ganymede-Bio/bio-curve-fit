import numpy as np
import pandas as pd
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

    params = list(model.get_params().values())
    assert np.isclose(params, TEST_PARAMS, rtol=0.4).all()  # type: ignore

    r2 = model.score(x_data, y_data)
    assert r2 > 0.995

    plot_curve(x_data, y_data, model)

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
    plot_curve(test_x, test_y, model)
    assert model.score(test_x, test_y) > 0.995  # type: ignore
    print(model.ULOD_y_, model.LLOD_y_)

    assert model.ULOD_y_ == 220006.8397685415
    assert model.LLOD_y_ == 798.7000577483678

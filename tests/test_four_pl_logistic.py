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

    params = model.get_params()
    assert np.isclose(params, TEST_PARAMS, rtol=0.4).all()  # type: ignore

    r2 = model.score(x_data, y_data)
    assert r2 > 0.95

    plot_curve(x_data, y_data, model)


y = pd.Series(
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

# x = pd.Series([
#     1550.000000,
#     1550.000000,
#     387.500000,
#     387.500000,
#     96.875000,
#     96.875000,
#     24.218750,
#     24.218750,
#     6.054688,
#     6.054688,
#     1.513672,
#     1.513672,
#     0.378417969,
#     0.378417969,
#     0.094604492,
#     0.094604492,
#     0.000000,
#     0.000000,
# ])

x = pd.Series(
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


# x = [
#     1550.000000,
#     1550.000000,
#     387.500000,
#     387.500000,
#     96.875000,
#     96.875000,
#     24.218750,
#     24.218750,
#     6.054688,
#     6.054688,
#     1.513672,
#     1.513672,
#     0.000000,
#     0.000000,
# ]

# y = [
#     64926,
#     62093,
#     13425,
#     13040,
#     3142,
#     3091,
#     1052,
#     834,
#     342,
#     278,
#     245,
#     250,
#     248,
#     190,
# ]


def test_umoja():
    model = FourPLLogistic().fit(
        x,
        y,
        weight_func=FourPLLogistic.inverse_variance_weight_function,
    )
    print("Params:", model.get_params())
    plot_curve(x, y, model)

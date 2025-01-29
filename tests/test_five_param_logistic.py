from dataclasses import asdict

import numpy as np

from bio_curve_fit.logistic import FiveParamLogistic, FourParamLogistic
from bio_curve_fit.plotting import plot_standard_curve, plot_standard_curve_figure

# set a seed for reproducibility
np.random.seed(42)


def test_fit():
    TEST_PARAMS = {"A": 1, "B": 1, "C": 2, "D": 3, "E": 0.4}

    x_data = np.logspace(0.00001, 7, 100, base=np.e)  # type: ignore
    # generate y-data based on the test parameters
    y_data = FiveParamLogistic._logistic_model(x_data, **TEST_PARAMS)
    y_data_with_noise = y_data + np.random.normal(0.0, 0.01 * y_data, len(y_data))

    model = FiveParamLogistic().fit(
        x_data,
        y_data_with_noise,
        weight_func=FiveParamLogistic.inverse_variance_weight_function,
    )

    # model should recover parameters used to generate the data
    fitted_params = model.get_params()

    for k, v in TEST_PARAMS.items():
        assert np.isclose(fitted_params[k], v, rtol=0.2)

    r2 = model.score(x_data, y_data)
    assert r2 > 0.995

    # test __repr__
    print(model)


known_concentrations = np.array(
    [
        0.01,
        0.01438449888287663,
        0.0206913808111479,
        0.029763514416313176,
        0.04281332398719394,
        0.06158482110660264,
        0.08858667904100823,
        0.12742749857031335,
        0.18329807108324356,
        0.26366508987303583,
        0.37926901907322497,
        0.5455594781168517,
        0.7847599703514611,
        1.1288378916846884,
        1.623776739188721,
        2.3357214690901213,
        3.359818286283781,
        4.832930238571752,
        6.951927961775605,
        10.0,
    ]
)

instrument_responses = np.array(
    [
        0.002069117990872958,
        0.001959532926455008,
        0.0019060618132396786,
        0.0020199075226226327,
        0.00232136833694934,
        0.002350039420066734,
        0.0034268951061987517,
        0.005976982077588455,
        0.01429016019812947,
        0.03919501065794866,
        0.11012486126580442,
        0.28788349523919826,
        0.6293705171187065,
        1.0942268483928872,
        1.5682314346793895,
        1.9810187452329917,
        2.3179681138726,
        2.5871131435953965,
        2.8003648496549087,
        2.968849253649377,
    ]
)


def test_fit2():
    model = FiveParamLogistic().fit(
        known_concentrations,
        instrument_responses,
        weight_func=FiveParamLogistic.inverse_variance_weight_function,
    )

    print(model.predict_inverse(0.1))
    assert model.score(known_concentrations, instrument_responses) > 0.995  # type: ignore
    print(model.ULOD_y_, model.LLOD_y_)

    assert np.isclose(model.ULOD_y_, 2.997, rtol=0.1)  # type: ignore
    assert np.isclose(model.LLOD_y_, 0.00213, rtol=0.1)  # type: ignore


def test_compare_4PL_to_5PL():
    four_param_model = FourParamLogistic().fit(
        known_concentrations,
        instrument_responses,
        weight_func=FourParamLogistic.inverse_variance_weight_function,
    )

    five_param_model = FiveParamLogistic().fit(
        known_concentrations,
        instrument_responses,
        weight_func=FiveParamLogistic.inverse_variance_weight_function,
    )

    four_param_model.fit(
        known_concentrations,
        instrument_responses,
        weight_func=FourParamLogistic.inverse_variance_weight_function,
    )
    five_param_model.fit(
        known_concentrations,
        instrument_responses,
        weight_func=FourParamLogistic.inverse_variance_weight_function,
    )

    four_param_model.score(known_concentrations, instrument_responses)
    five_param_model.score(known_concentrations, instrument_responses)

    assert four_param_model.score(
        known_concentrations, instrument_responses
    ) < five_param_model.score(known_concentrations, instrument_responses)


def test_std_dev():
    model = FiveParamLogistic().fit(
        known_concentrations,
        instrument_responses,
    )
    ci = model.predict_confidence_band(known_concentrations)
    pi = model.predict_prediction_band(known_concentrations, instrument_responses)

    print(ci)
    print(pi)

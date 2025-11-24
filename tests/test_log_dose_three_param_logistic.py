import numpy as np
import pytest

from bio_curve_fit.logistic import LogDoseFourParamLogistic, LogDoseThreeParamLogistic

# set a seed for reproducibility
np.random.seed(42)


def test_fit_and_predict():
    """Test basic fitting and prediction functionality."""
    np.random.seed(42)
    TEST_PARAMS = {"A": 1, "D": 3, "C": 2}

    # Generate test data
    x_data = np.linspace(-2, 6, 50)
    y_data = LogDoseThreeParamLogistic._logistic_model(x_data, **TEST_PARAMS)
    y_data_with_noise = y_data + np.random.normal(0, 0.01, len(y_data))

    # Test that unfitted model raises error
    model = LogDoseThreeParamLogistic()
    with pytest.raises(ValueError):
        model.predict(x_data)

    # Fit the model
    model.fit(x_data, y_data_with_noise)

    # Check that parameters are recovered reasonably well
    fitted_params = model.get_params()
    for key, value in TEST_PARAMS.items():
        assert np.isclose(fitted_params[key], value, rtol=0.1)

    # Test prediction
    predictions = model.predict(x_data)
    assert len(predictions) == len(x_data)

    # Test R² score
    r2 = model.score(x_data, y_data)
    assert r2 > 0.99


def test_jacobian():
    """Test Jacobian calculation."""
    # Create and fit a model first
    model = LogDoseThreeParamLogistic()
    model.A = 1
    model.D = 3
    model.C = 2

    x_data = np.array([1.0, 2.0, 3.0])
    J = model.jacobian(x_data)

    # Check shape
    assert J.shape == (3, 3)  # 3 data points, 3 parameters

    # Check that jacobian values are reasonable (not NaN or inf)
    assert np.all(np.isfinite(J))


def test_confidence_bands():
    """Test confidence band calculation."""
    np.random.seed(42)
    TEST_PARAMS = {"A": 1, "D": 3, "C": 2}

    x_data = np.linspace(-2, 6, 50)
    y_data = LogDoseThreeParamLogistic._logistic_model(x_data, **TEST_PARAMS)
    y_data_with_noise = y_data + np.random.normal(0, 0.01, len(y_data))

    model = LogDoseThreeParamLogistic()
    model.fit(x_data, y_data_with_noise)

    # Test confidence bands
    cb = model.predict_confidence_band(x_data)
    assert len(cb) == len(x_data)
    assert np.all(cb >= 0)  # Confidence bands should be positive

    # Test prediction bands
    pb = model.predict_prediction_band(x_data, y_data_with_noise)
    assert len(pb) == len(x_data)
    assert np.all(pb >= 0)  # Prediction bands should be positive
    assert np.all(pb >= cb)  # Prediction bands should be wider than confidence bands


def test_real_data_example():
    """Test fitting with real data example verifying against known values from Prism"""
    # Log concentrations (assuming already in log scale)
    log_concentrations = np.array(
        [
            -4.602,
            -5.301,
            -6.000,
            -6.699,
            -7.398,
            -8.097,
            -4.602,
            -5.301,
            -6.000,
            -6.699,
            -4.602,
            -5.301,
            -6.000,
            -6.699,
            -8.097,
        ]
    )

    # Responses following log-dose 3PL pattern
    responses = np.array(
        [
            119.2,
            117.6,
            107.3,
            75.6,
            66.6,
            8.3,
            113.8,
            115.2,
            101.8,
            76.7,
            115.8,
            113.9,
            95.5,
            16.8,
            -57.1,
        ]
    )

    model = LogDoseThreeParamLogistic()
    model.fit(log_concentrations, responses)

    # Check that fit matches expected values
    assert model.A == pytest.approx(-22.04, rel=0.001)
    assert model.D == pytest.approx(115.0, rel=0.001)
    assert model.C == pytest.approx(-6.995, rel=0.001)


# LogDoseFourParamLogistic tests


def test_logdose_4pl_fit_and_predict():
    """Test basic fitting and prediction functionality for LogDose 4PL."""
    np.random.seed(42)
    TEST_PARAMS = {"A": 1, "B": 1.5, "C": 2, "D": 3}

    # Generate test data
    x_data = np.linspace(-2, 6, 50)
    y_data = LogDoseFourParamLogistic._logistic_model(x_data, **TEST_PARAMS)
    y_data_with_noise = y_data + np.random.normal(0, 0.01, len(y_data))

    # Test that unfitted model raises error
    model = LogDoseFourParamLogistic()
    with pytest.raises(ValueError):
        model.predict(x_data)

    # Fit the model
    model.fit(x_data, y_data_with_noise)

    # Check that parameters are recovered reasonably well
    fitted_params = model.get_params()
    for key, value in TEST_PARAMS.items():
        assert np.isclose(fitted_params[key], value, rtol=0.1)

    # Test prediction
    predictions = model.predict(x_data)
    assert len(predictions) == len(x_data)

    # Test R² score
    r2 = model.score(x_data, y_data)
    assert r2 > 0.99


def test_logdose_4pl_predict_inverse():
    """Test inverse prediction for LogDose 4PL."""
    np.random.seed(42)
    TEST_PARAMS = {"A": 1, "B": 1.5, "C": 2, "D": 3}

    model = LogDoseFourParamLogistic(**TEST_PARAMS)

    # Test inverse prediction
    y_values = np.array([1.5, 2.0, 2.5])
    x_predictions = model.predict_inverse(y_values)

    # Verify round-trip: predict_inverse -> predict should give back y_values
    y_roundtrip = model.predict(x_predictions)
    assert np.allclose(y_values, y_roundtrip, rtol=0.01)


def test_logdose_4pl_with_fixed_b_matches_3pl():
    """Test that LogDose 4PL with B=1 matches LogDose 3PL."""
    np.random.seed(42)
    TEST_PARAMS_3PL = {"A": 1, "D": 3, "C": 2}

    x_data = np.linspace(-2, 6, 50)
    y_data = LogDoseThreeParamLogistic._logistic_model(x_data, **TEST_PARAMS_3PL)
    y_data_with_noise = y_data + np.random.normal(0, 0.01, len(y_data))

    # Fit LogDose 3PL
    model_3pl = LogDoseThreeParamLogistic()
    model_3pl.fit(x_data, y_data_with_noise)

    # Fit LogDose 4PL with B=1 fixed
    model_4pl = LogDoseFourParamLogistic(B=1)
    model_4pl.fit(x_data, y_data_with_noise)

    # Parameters should be close
    assert np.isclose(model_3pl.A, model_4pl.A, rtol=0.01)
    assert model_4pl.B == 1.0  # Should remain fixed
    assert np.isclose(model_3pl.C, model_4pl.C, rtol=0.01)
    assert np.isclose(model_3pl.D, model_4pl.D, rtol=0.01)

    # Predictions should be nearly identical
    predictions_3pl = model_3pl.predict(x_data)
    predictions_4pl = model_4pl.predict(x_data)
    assert np.allclose(predictions_3pl, predictions_4pl, rtol=0.01)

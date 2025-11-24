"""Base class for logistic models."""

from abc import ABC, abstractmethod
from inspect import signature
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit  # type: ignore
from scipy.stats import t  # type: ignore
from sklearn.base import RegressorMixin  # type: ignore

from .base import BaseStandardCurve


class LogisticRegression(RegressorMixin, BaseStandardCurve, ABC):
    """
    Abstract base class for logistic regression model and associated fitting.

    Attributes
    ----------
    LLOD
    ULOD
    ULOD_y
    LLOD_y
    slope_direction_positive
    slope_guess_num_points_to_use

    Methods
    -------
    inverse_variance_weight_function(data):
        Weight function for weighting residuals by 1/y^2 in `scipy.optimize.curve_fit`.

    _calculate_lod_replicate_variance:
        Calculate upper and lower limit of detection for provided dataset.

    _logistic_model:
        Functional form for logistic regression model. Implemented by subclasses

    predict:
        Applies fit model to provided samples. Implemented by subclasses.

    generate_initial_param_values
        Generate initial parameter values for model. Implemented by subclasses.

    """

    def __init__(
        self,
        LLOD=None,
        ULOD=None,
        ULOD_y=None,
        LLOD_y=None,
        slope_direction_positive: Optional[bool] = None,
        slope_guess_num_points_to_use: int = 3,
    ):
        self.cov_ = None
        self.n_samples_ = (
            None  # Store number of training data points for DOF calculation
        )
        # Estimated Limits of Detection for response signal
        self.LLOD_y_ = LLOD_y
        self.ULOD_y_ = ULOD_y
        # Estimated Limits of Detection for concentration
        self.LLOD_ = LLOD
        self.ULOD_ = ULOD
        self.slope_direction_positive = slope_direction_positive
        self.slope_guess_num_points_to_use = slope_guess_num_points_to_use

    def _calculate_lod_replicate_variance(
        self,
        x_data,
        y_data,
        lower_std_dev_multiplier: float = 2.5,
        upper_std_dev_multiplier: float = 0.0,
    ):
        """Calculate the Lower and Upper Limits of Detection (LLOD and ULOD).

        Uses variance of replicate max and min concentration standards. It ignores
        zero concentration standards. If there are no replicates, the standard
        deviation zero. Possible TODO: sometimes a minimum variance is used in other software.

        In the notation below we assume the response signal is the Y-axis and the
        concentration is the X-axis.

        Example: Two replicates of the lowest concentration standard (conc=1.0 pg/ml)
        have standard deviation of 100 across their responses. LLOD will be `model.predict
        (1.0) + 100 * 2.5` where 2.5 is the `lower_std_dev_multiplier` parameter.

        Parameters
        ----------
        x_data: x data points
        y_data: y data points
        lower_std_dev_multiplier: Multiplier how many standard deviations to add to the lowest calibration point to get the LLOD.
        upper_std_dev_multiplier: Multiplier how many standard deviations to subtract from the highest calibration point to get the ULOD.

        Returns
        -------
        tuple[float, float, float, float]:
            the LLOD and ULOD, and the corresponding x-values.
        """
        x_indexed_y_data = pd.DataFrame({"x": x_data, "y": y_data}).set_index("x")
        # remove zeros from x_data (but keep negative values for log-scale data)
        x_indexed_y_data = x_indexed_y_data[x_indexed_y_data.index.to_numpy() != 0]
        x_min = np.min(x_indexed_y_data.index.to_numpy())
        x_max = np.max(x_indexed_y_data.index.to_numpy())
        bottom_std_dev = x_indexed_y_data.loc[x_min, "y"].std()
        top_std_dev = x_indexed_y_data.loc[x_max, "y"].std()

        # Calculate LLOD and ULOD of RESPONSE SIGNAL
        llod = self.predict(x_min) + (lower_std_dev_multiplier * bottom_std_dev)
        ulod = self.predict(x_max) - (upper_std_dev_multiplier * top_std_dev)

        # Calculate the limits of detection for CONCENTRATION
        llod_x = self.predict_inverse(llod)
        ulod_x = self.predict_inverse(ulod)
        return llod_x, ulod_x, llod, ulod

    def _check_fit_params(self):
        for v in self.get_params().values():
            if v is None:
                raise ValueError(
                    "Model is not fit yet. Please call 'fit' with appropriate data"
                    " or initialize the model object with non-null parameters."
                )

    def _calculate_dof_and_t_critical(self, n_params, n_data_points=None, alpha=0.05):
        """Calculate degrees of freedom and critical t-value for confidence/prediction bands.

        Parameters
        ----------
        n_params : int
            Number of parameters in the model
        n_data_points : int, optional
            Number of data points. If None, uses self.n_data_points_
        alpha : float, default=0.05
            Significance level for confidence/prediction interval

        Returns
        -------
        tuple[int, float]
            Degrees of freedom and critical t-value
        """
        if n_data_points is None:
            n_data_points = self.n_samples_

        dof = n_data_points - n_params
        t_crit = t.ppf(1.0 - alpha / 2.0, dof)

        return dof, t_crit

    def fit(
        self,
        x_data,
        y_data,
        weight_func=None,
        LOD_func=None,
        initial_param_values=None,
        **kwargs,
    ):
        """
        Fit the associated Logistic model with support for fixed parameters.

        If any model parameters are set to non-None values during initialization,
        those parameters will be held constant during fitting.

        Parameters
        ----------
        x_data: array-like
            x data points
        y_data: array-like
            y data points
        weight_func: callable, optional
            Function that calculates weights from y_data. This is
            passed into the `curve_fit` function where the function minimized is `sum
            ((r / weight_func(y_data)) ** 2)` where r is the residuals.
            Thus for a typical 1/y^2 weighting, `weight_func` should be `lambda
            y_data: y_data`
        LOD_func: callable, optional
            Function to calculate limits of detection. If None, uses
            _calculate_lod_replicate_variance
        initial_param_values: array-like, optional
            Initial parameter guesses. If None, uses generate_initial_param_values
        **kwargs: dict
            Additional keyword arguments passed to scipy.optimize.curve_fit
        """
        x_data = np.array(x_data, dtype=np.float64)
        y_data = np.array(y_data, dtype=np.float64)

        if LOD_func is None:
            LOD_func = self._calculate_lod_replicate_variance

        # Get parameter names from the _logistic_model signature
        function_args = list(signature(self._logistic_model).parameters.keys())
        function_args.remove("x")

        # Determine which parameters are fixed (non-None) and which are free
        fixed_params = {}
        free_param_names = []

        for param_name in function_args:
            param_value = getattr(self, param_name, None)
            if param_value is not None:
                fixed_params[param_name] = param_value
            else:
                free_param_names.append(param_name)

        # If all parameters are fixed, no fitting needed
        if len(free_param_names) == 0:
            self.LLOD_, self.ULOD_, self.LLOD_y_, self.ULOD_y_ = LOD_func(
                x_data, y_data
            )
            self._free_param_names = []
            self._has_fixed_params = True
            return self

        # Create a wrapper function that only optimizes free parameters
        def constrained_model(x, *free_params):
            # Reconstruct full parameter set
            param_dict = fixed_params.copy()
            for name, value in zip(free_param_names, free_params):
                param_dict[name] = value
            # Call _logistic_model with parameters in the correct order
            param_values = [param_dict[name] for name in function_args]
            return self._logistic_model(x, *param_values)

        # Generate initial values for free parameters only
        if not initial_param_values:
            full_initial = self.generate_initial_param_values(x_data, y_data)
            # Map parameter names to their indices in the full initial array
            param_name_to_idx = {name: idx for idx, name in enumerate(function_args)}
            initial_param_values = [
                full_initial[param_name_to_idx[name]] for name in free_param_names
            ]

        # Setup weights
        absolute_sigma = False
        weights = None
        if weight_func is not None:
            weights = weight_func(y_data)
            absolute_sigma = True

        # Setup curve_fit arguments
        curve_fit_kwargs = {
            "f": constrained_model if fixed_params else self._logistic_model,
            "xdata": x_data,
            "ydata": y_data,
            "p0": initial_param_values,
            "maxfev": 10000,
            "sigma": weights,
            "absolute_sigma": absolute_sigma,
        }

        # Overwrite parameters with any kwargs passed in
        for k, v in kwargs.items():
            curve_fit_kwargs[k] = v

        # Perform the curve fit
        fitted_params, cov = curve_fit(**curve_fit_kwargs)

        # Set the fitted parameters
        if fixed_params:
            # Only set free parameters
            for name, value in zip(free_param_names, fitted_params):
                setattr(self, name, value)
        else:
            # Set all parameters
            params_dict = {k: v for k, v in zip(function_args, fitted_params)}
            self.set_params(**params_dict)

        # Store information about which parameters were fixed
        self._free_param_names = free_param_names
        self._has_fixed_params = len(free_param_names) < len(function_args)

        self.cov_ = cov
        self.n_samples_ = len(x_data)  # Store for DOF calculation
        self.LLOD_, self.ULOD_, self.LLOD_y_, self.ULOD_y_ = LOD_func(x_data, y_data)

        return self

    @staticmethod
    def inverse_variance_weight_function(y_data):
        """
        Weight function for weighting residuals by 1/y^2 in `scipy.optimize.curve_fit`.

        Parameters
        ----------
            y_data: y data points
        """
        # To avoid division by zero, add a small constant to y_data.
        return y_data + np.finfo(float).eps

    @staticmethod
    @abstractmethod
    def _logistic_model(x, *args):
        pass

    @abstractmethod
    def predict(self, x_data):
        """Override this method in subclasses."""
        pass

    @abstractmethod
    def generate_initial_param_values(self, x_data, y_data):
        """Override this method in subclasses."""
        pass

    @abstractmethod
    def jacobian(self, x_data):
        """Calculate the Jacobian matrix for the model."""
        pass

    def _predict_bands(self, x_data, alpha=0.05, y_data=None):
        """
        Calculate confidence or prediction bands.

        Parameters
        ----------
        x_data : array-like
            Input data points for which to calculate bands
        alpha : float, default=0.05
            Significance level for interval (0.05 for 95% interval)
        y_data : array-like, optional
            Training response data. If provided, calculates prediction bands.
            If None, calculates confidence bands.

        Returns
        -------
        np.ndarray
            Half-width of band at each x_data point
        """
        if self.cov_ is None:
            raise ValueError(
                "Covariance matrix is not available. Please call 'fit' with appropriate data."
            )

        # Calculate degrees of freedom and critical t-value
        n_params = len(self.get_params())
        n_data_points = len(y_data) if y_data is not None else None
        dof, t_crit = self._calculate_dof_and_t_critical(
            n_params, n_data_points=n_data_points, alpha=alpha
        )

        # Calculate confidence variance
        J = self.jacobian(x_data)
        conf_var = np.sum((J @ self.cov_) * J, axis=1)

        # For prediction bands, add residual variance
        if y_data is not None:
            residuals = y_data - self.predict(x_data)
            mse = np.sum(residuals**2) / dof
            pred_var = conf_var + mse
        else:
            pred_var = conf_var

        return t_crit * np.sqrt(pred_var)

    def predict_confidence_band(self, x_data, alpha=0.05):
        """
        Predict confidence bands of data points.

        Parameters
        ----------
        x_data : array-like
            Input data points for which to calculate confidence bands
        alpha : float, default=0.05
            Significance level for confidence interval (0.05 for 95% confidence)

        Returns
        -------
        np.ndarray
            Half-width of confidence band at each x_data point
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        return self._predict_bands(x_data, alpha=alpha)

    def predict_prediction_band(self, x_data, y_data, alpha=0.05):
        """Predict prediction bands of data points.

        Parameters
        ----------
        x_data : array-like
            Input data points for which to calculate prediction bands
        y_data : array-like
            Training response data used to calculate residual variance
        alpha : float, default=0.05
            Significance level for prediction interval (0.05 for 95% prediction)

        Returns
        -------
        np.ndarray
            Half-width of prediction band at each x_data point

        Notes
        -----
        Prediction bands include both parameter uncertainty (from confidence bands)
        and residual variance from the model fit.
        """
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
        return self._predict_bands(x_data, alpha=alpha, y_data=y_data)


class FiveParamLogistic(LogisticRegression):
    r"""

    Five Parameter Logistic (5PL) model.

    The 5PL model is a sigmoidal curve that is defined by the following equation:

    .. math::
        f(x) = D + \frac{A - D}{\left(1 + \left(\frac{x}{C}\right)^B\right)^S}

    Where:
        - A is the minimum asymptote
        - B is the Hill's slope
        - C is the inflection point (EC50)
        - D is the maximum asymptote
        - E is the asymmetry factor

    The 5PL model is commonly used in bioassays, such as ELISA, where the response signal is
    proportional to the concentration of the analyte being measured. It's used to
    fit a standard curve, which is a plot of the response signal against the concentration of
    known standards. The 5PL accounts for asymmetry in response and can provide better performance
    for skewed data when compared to the 4PL. The standard curve is then used to estimate the concentration of unknown
    samples based on their response signal.
    """

    def __init__(self, A=None, B=None, C=None, D=None, E=None, **kwargs):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        super().__init__(**kwargs)

    @staticmethod
    def _logistic_model(x, A, B, C, D, E):
        """5 Parameter Logistic (5PL) model."""
        # For addressing fractional powers of negative numbers
        # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
        z = (np.sign(x / C) * np.abs(x / C)) ** B
        denominator = (1.0 + z) ** E

        return ((A - D) / denominator) + D

    @staticmethod
    def _tangent_line_at_midpoint(x, A, B, C, D, E):
        """Line with slope = (Derivative of the 5PL curve evaluated at C) and passing through the point C."""
        term1 = D + (A - D) / 2**E
        term2 = (A - D) * E * B / (2 ** (E + 1) * C)
        term3 = x - C
        return term1 - term2 * term3

    def tangent_line_at_midpoint(self, x):
        """Return y value of the tangent line at the inflection point (C) of the 4PL curve.

        This is an alternate way to define the linear range of the curve, that is, the part of the curve where it is approximately linear.
        """
        self._check_fit_params()
        return self._tangent_line_at_midpoint(
            x,
            self.A,
            self.B,
            self.C,
            self.D,
            self.E,
        )

    @staticmethod
    def _tangent_line_at_arbitrary_point(x, g, A, B, C, D, E):
        """Return y value of line with slope = (Derivative of the 5PL curve evaluated at g) and passing through the point (g, f(g))."""
        derivative_at_g = (-1 * (A - D) * E * B * (g / C) ** (B - 1)) / (
            C * (1 + (g / C) ** B) ** (E + 1)
        )

        line_with_slope_at_g = derivative_at_g * (
            x - g
        ) + FiveParamLogistic._logistic_model(g, A, B, C, D, E)

        return line_with_slope_at_g

    def tangent_line_at_arbitrary_point(self, x, g):
        """Return the f(x) where f is the line tangent to the 5PL curve at the point g."""
        self._check_fit_params()
        return FiveParamLogistic._tangent_line_at_arbitrary_point(
            x,
            g,
            self.A,
            self.B,
            self.C,
            self.D,
            self.E,
        )

    def generate_initial_param_values(self, x_data, y_data):
        """Generate an initial guess for the parameters of the 5PL model based on the data.

        Uses fixed parameter values if they are already set (not None), otherwise generates
        reasonable guesses from the data.
        """
        x_data = np.float64(x_data)
        y_data = np.float64(y_data)
        df_data = pd.DataFrame({"x": x_data, "y": y_data})
        df_data.sort_values(by="x", inplace=True)

        # Use fixed value if parameter is already set, otherwise generate initial guess
        initial_A = self.A if self.A is not None else np.min(y_data)  # type: ignore

        if self.B is not None:
            initial_B = self.B
        elif self.slope_direction_positive is not None:
            initial_B = 1.0 if self.slope_direction_positive else -1.0
        else:
            # type: ignore
            initial_B = (
                1.0
                if np.mean(
                    df_data.iloc[
                        : min(self.slope_guess_num_points_to_use, len(df_data))
                    ][  # type: ignore
                        "y"
                    ]
                )
                < np.mean(
                    df_data.iloc[
                        -min(self.slope_guess_num_points_to_use, len(df_data)) :
                    ][  # type: ignore
                        "y"
                    ]
                )
                else -1.0
            )

        initial_C = self.C if self.C is not None else np.mean(x_data)  # type: ignore
        initial_D = self.D if self.D is not None else np.max(y_data)  # type: ignore
        initial_E = self.E if self.E is not None else 1.0

        initial_guess = [initial_A, initial_B, initial_C, initial_D, initial_E]

        return initial_guess

    def jacobian(self, x_data):
        """Jacobian matrix of the 5PL function with respect to A, B, C, D, E."""
        self._check_fit_params()
        z = (x_data / self.C) ** self.B

        partial_A = 1.0 / (1.0 + z) ** self.E

        partial_B_num = (
            -(self.A - self.D)  # type: ignore
            * self.E
            * z
            * np.log(
                np.maximum(
                    np.array(x_data, dtype=float) / self.C,
                    np.finfo(float).eps * np.ones_like(x_data),
                )
            )
        )
        partial_B_denom = (1 + z) ** (self.E + 1)  # type: ignore
        partial_B = partial_B_num / partial_B_denom

        partial_C_num = -(self.A - self.D) * self.E * self.B * z  # type: ignore
        partial_C_denom = self.C * (1 + z) ** (self.E + 1)  # type: ignore
        partial_C = partial_C_num / partial_C_denom

        partial_D = 1.0 - 1.0 / (1.0 + z) ** self.E

        partial_E_num = -(self.A - self.D) * np.log(1 + z)  # type: ignore
        partial_E_denom = (1 + z) ** self.E

        partial_E = partial_E_num / partial_E_denom

        # Jacobian matrix
        J = np.array([partial_A, partial_B, partial_C, partial_D, partial_E]).T
        return J

    def predict_inverse(
        self,
        y: Union[float, int, np.ndarray, Iterable[float]],
        enforce_limits: bool = True,
    ):
        """Inverse 5 Parameter Logistic (5PL) model.

        Used for calculating the x-value for a given y-value.
        Usually, standard curves are fitted using concentration as x-values and response as
        y-values, so that variance in response is modeled for a given known concentration.
        But for samples of unknown concentration, we want to get the concentration as given
        response, which is what this function does.

        Parameters
        ----------
        y: float or iterable
            The response value for which the corresponding x-value will be calculated.
        enforce_limits: bool
            If True, return np.nan for y-values above the maximum asymptote (D) of the curve, and 0 for y-values below the minimum asymptote (A) of the curve.

        """
        self._check_fit_params()
        if isinstance(y, list):
            y = np.array(y, dtype=float)

        z = (self.A - self.D) / (y - self.D)  # type: ignore

        term1 = (np.sign(z) * np.abs(z)) ** (1 / self.E) - 1  # type: ignore

        # For addressing fractional powers of negative numbers, np.sign * abs pattern handles negative values
        # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
        x = self.C * (np.sign(term1) * np.abs(term1) ** (1 / self.B))  # type: ignore

        if enforce_limits:
            if isinstance(y, (np.ndarray, pd.Series)):
                x[y > self.D] = np.nan  # type: ignore
                x[y < self.A] = 0  # type: ignore
            elif isinstance(y, (int, float)):
                if y > self.D:  # type: ignore
                    return np.nan
                elif y < self.A:  # type: ignore
                    return 0
        return x

    def predict(self, x_data):
        """Predict y-values using the 5PL model.

        Parameters
        ----------
            x_data (iterable): x data points

        Returns
        -------
            iterable: y data points
        """
        self._check_fit_params()
        return self._logistic_model(
            x_data,
            self.A,
            self.B,
            self.C,
            self.D,
            self.E,
        )


class FourParamLogistic(FiveParamLogistic):
    """Implementation of the 4 Parameter Logistic (4PL) model.

    The 4PL is a special case of the 5PL model with E=1 (symmetric curve).
    Supports parameter constraints by fixing specific parameters during initialization.
    For example, to create a 3PL model with fixed Hill's slope:
        model = FourParamLogistic(B=1.0)

    Parameters
    ----------
    A : float, optional
        Minimum asymptote. If provided, this parameter will be fixed during fitting.
    B : float, optional
        Hill's slope. If provided, this parameter will be fixed during fitting.
    C : float, optional
        Inflection point (EC50). If provided, this parameter will be fixed during fitting.
    D : float, optional
        Maximum asymptote. If provided, this parameter will be fixed during fitting.
    """

    def __init__(self, A=None, B=None, C=None, D=None, **kwargs):
        # FourPL is FivePL with E=1 (symmetric curve)
        super().__init__(A=A, B=B, C=C, D=D, E=1, **kwargs)

    @staticmethod
    def _logistic_model(x, A, B, C, D, E=1):
        """4 Parameter Logistic (4PL) model.

        E parameter is accepted but ignored (always treated as 1) for compatibility
        with parent class predict() method.
        """
        return FiveParamLogistic._logistic_model(x, A, B, C, D, E=1)

    def semi_log_linear_range_of_response(self) -> Tuple[float, float]:
        """
        Return the response range where the curve is approximately linear in a semi-log plot.

        That is, it returns the lower and upper limit y-values where the curve will "look" linear
        when plotted on a log scaled x-axis (usually concentration) and a linear y-axis (usually response).

        Follows Sebaugh, Jeanne & McCray, P.. (2003). Defining the linear portion of a sigmoid-shaped curve: Bend points. Pharmaceutical Statistics - PHARM STAT. 2. 167-174. 10.1002/pst.62. See the pdf in the references folder in this repo, or https://www.researchgate.net/publication/246918700_Defining_the_linear_portion_of_a_sigmoid-shaped_curve_Bend_points


        """
        self._check_fit_params()
        K = 4.680498579  # Magic number from the paper
        A = self.A
        D = self.D
        y_bend_lower = (A - D) / (1 + 1 / K) + D  # type: ignore
        y_bend_upper = (A - D) / (1 + K) + D  # type: ignore
        return y_bend_lower, y_bend_upper

    @staticmethod
    def _tangent_line_at_midpoint(x, A, B, C, D):
        """Line with slope = (Derivative of the 4PL curve evaluated at C) and passing through the point C."""
        return -(A - D) * B / (4 * C) * (x - C) + (A + D) / 2

    @staticmethod
    def _tangent_line_at_arbitrary_point(x, g, A, B, C, D):
        """Return y value of line with slope = (Derivative of the 4PL curve evaluated at g) and passing through the point (g, f(g))."""
        derivative_at_g = (
            -1 * (A - D) * B * (g / C) ** (B - 1) / (C * (1 + (g / C) ** B) ** 2)
        )

        line_with_slope_at_g = derivative_at_g * (
            x - g
        ) + FourParamLogistic._logistic_model(g, A, B, C, D)
        return line_with_slope_at_g

    def tangent_line_at_arbitrary_point(self, x, g):
        """Return the f(x) where f is the line tangent to the 4PL curve at the point g."""
        self._check_fit_params()
        return FourParamLogistic._tangent_line_at_arbitrary_point(
            x, g, self.A, self.B, self.C, self.D
        )

    def tangent_line_at_midpoint(self, x):
        """Return the f(x) where f is the line tangent to the 4PL curve at the point g."""
        self._check_fit_params()
        return FourParamLogistic._tangent_line_at_midpoint(
            x, self.A, self.B, self.C, self.D
        )

    def jacobian(self, x_data):
        """Jacobian matrix of the 4PL function with respect to A, B, C, D.

        Returns a 4-column Jacobian (without E) since E is fixed at 1 for 4PL.
        """
        self._check_fit_params()
        z = (x_data / self.C) ** self.B

        partial_A = 1.0 / (1.0 + z)
        partial_B = -(
            z
            * (self.A - self.D)  # type: ignore
            * np.log(np.maximum(x_data / self.C, np.finfo(float).eps))  # type: ignore
        ) / (  # type: ignore
            (1.0 + z) ** 2
        )
        partial_C = (self.B * z * (self.A - self.D)) / (self.C * (1.0 + z) ** 2)  # type: ignore
        partial_D = 1.0 - 1.0 / (1.0 + z)

        # Jacobian matrix (4 columns, without E since E=1 is fixed)
        J = np.array([partial_A, partial_B, partial_C, partial_D]).T
        return J


class LogDoseFourParamLogistic(FiveParamLogistic):
    """Implementation of the Log-Dose 4-Parameter Logistic model.

    The Log-Dose 4PL is a variant of the 5PL model that uses base-10 exponential
    form instead of power-law. It's defined by:
    Y = A + (D - A) / (1 + 10^((C - X) * B))

    This is commonly used in pharmacology and biochemistry, where:
    - A: minimum asymptote (Bottom)
    - B: Hill slope
    - C: log10 of the concentration at half-maximal response (LogEC50)
    - D: maximum asymptote (Top)

    Parameters
    ----------
    A : float, optional
        Minimum asymptote parameter. Set during fitting if not provided.
    B : float, optional
        Hill slope parameter. Set during fitting if not provided.
    C : float, optional
        Log10 of EC50 concentration parameter. Set during fitting if not provided.
    D : float, optional
        Maximum asymptote parameter. Set during fitting if not provided.
    """

    def __init__(self, A=None, B=None, C=None, D=None, **kwargs):
        # LogDose4PL is like 5PL with E=1 but uses exponential form
        super().__init__(A=A, B=B, C=C, D=D, E=1, **kwargs)

    @staticmethod
    def _logistic_model(x, A, B, C, D, E=1):
        """Log-Dose 4PL model: Y = A + (D-A)/(1+10^((C-X)*B)).

        E parameter is accepted but ignored (always treated as 1) for compatibility
        with parent class.
        """
        return A + (D - A) / (1 + 10 ** ((C - x) * B))

    def predict_inverse(self, y):
        """Inverse Log-Dose 4PL model.

        Overrides parent to use log-dose specific inverse formula.
        """
        self._check_fit_params()
        if isinstance(y, list):
            y = np.array(y, dtype=float)

        # X = C - log10((D-y)/(y-A))/B
        ratio = (self.D - y) / (y - self.A)  # type: ignore
        x = self.C - np.log10(ratio) / self.B  # type: ignore
        return x


class LogDoseThreeParamLogistic(LogDoseFourParamLogistic):
    """Implementation of the Log-Dose 3-Parameter Logistic model.

    The Log-Dose 3PL model is a special case of the Log-Dose 4PL model with B=1 (Hill slope = 1).

    The model is defined by:
    Y = A + (D - A) / (1 + 10^((C - X)))

    This is a 3-parameter logistic model commonly used in pharmacology and biochemistry, where:
    - A: minimum asymptote (Bottom)
    - D: maximum asymptote (Top)
    - C: log10 of the concentration at half-maximal response (LogEC50)

    The Hill slope is fixed at 1.

    Parameters
    ----------
    A : float, optional
        Minimum asymptote parameter. Set during fitting if not provided.
    D : float, optional
        Maximum asymptote parameter. Set during fitting if not provided.
    C : float, optional
        Log10 of EC50 concentration parameter. Set during fitting if not provided.
    """

    def __init__(self, A=None, D=None, C=None, **kwargs):
        # LogDose3PL is LogDose4PL with B=1 (Hill slope = 1)
        super().__init__(A=A, B=1, C=C, D=D, **kwargs)

    @staticmethod
    def _logistic_model(x, A, B=1, C=None, D=None, E=1):
        """Log-Dose 3PL model: Y = A + (D-A)/(1+10^((C-X))).

        B and E parameters are accepted but ignored (always treated as 1) for compatibility
        with parent class. Can be called with (x, A, D, C) for backward compatibility.
        """
        return LogDoseFourParamLogistic._logistic_model(x, A, B=1, C=C, D=D, E=1)

    def jacobian(self, x_data):
        """Jacobian matrix of the Log-Dose 3PL function with respect to A, D, C."""
        self._check_fit_params()
        exp_term = 10 ** (self.C - x_data)
        denominator = (1 + exp_term) ** 2

        # ∂Y/∂A = 1 - 1/(1 + 10^(C-x)) = 10^(C-x)/(1 + 10^(C-x))
        partial_A = exp_term / (1 + exp_term)

        # ∂Y/∂D = 1/(1 + 10^(C-x))
        partial_D = 1 / (1 + exp_term)

        # ∂Y/∂C = (D-A) * ln(10) * 10^(C-x) / (1 + 10^(C-x))^2
        partial_C = (self.D - self.A) * np.log(10) * exp_term / denominator

        # Jacobian matrix
        J = np.array([partial_A, partial_D, partial_C]).T
        return J

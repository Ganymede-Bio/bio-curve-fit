"""Base class for logistic models."""

from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import RegressorMixin

from .base import BaseStandardCurve


class FourPLLogistic(RegressorMixin, BaseStandardCurve):
    r"""

    Four Parameter Logistic (4PL) model.

    The 4PL model is a sigmoidal curve that is defined by the following equation:

    .. math::
        f(x) = \\frac{A - D}{1 + (\\frac{x}{C})^B} + D

    Where:
        - A is the minimum asymptote
        - B is the Hill's slope
        - C is the inflection point (EC50)
        - D is the maximum asymptote

    The 4PL model is commonly used in bioassays, such as ELISA, where the response signal is
    proportional to the concentration of the analyte being measured. The 4PL model is used to
    fit a standard curve, which is a plot of the response signal against the concentration of
    known standards. The standard curve is then used to estimate the concentration of unknown
    samples based on their response signal.
    """

    def __init__(
        self,
        A=None,
        B=None,
        C=None,
        D=None,
        LLOD=None,
        ULOD=None,
        ULOD_y=None,
        LLOD_y=None,
        slope_direction_positive: Optional[bool] = None,
        slope_guess_num_points_to_use: int = 3,
    ):
        # A is the minimum asymptote
        self.A_ = A
        # B is the Hill's slope
        self.B_ = B
        # C is the inflection point (EC50)
        self.C_ = C
        # D is the maximum asymptote
        self.D_ = D
        self.cov_ = None
        # Initial guesses used when fitting the curve
        self.guess_A_ = None
        self.guess_B_ = None
        self.guess_C_ = None
        self.guess_D_ = None
        # Estimated Limits of Detection for response signal
        self.LLOD_y_ = LLOD_y
        self.ULOD_y_ = ULOD_y
        # Estimated Limits of Detection for concentration
        self.LLOD_ = LLOD
        self.ULOD_ = ULOD
        self.slope_direction_positive = slope_direction_positive
        self.slope_guess_num_points_to_use = slope_guess_num_points_to_use

    def _check_fit(self):
        if self.A_ is None or self.B_ is None or self.C_ is None or self.D_ is None:
            raise Exception(
                "Model is not fit yet. Please call 'fit' with appropriate data"
                " or initialize the model object with non-null parameters."
            )

    def semi_log_linear_range_of_response(self) -> Tuple[float, float]:
        """
        Return the response range where the curve is approximately linear in a semi-log plot.

        That is, it returns the lower and upper limit y-values where the curve will "look" linear
        when plotted on a log scaled x-axis (usually concentration) and a linear y-axis (usually response).

        Follows Sebaugh, Jeanne & McCray, P.. (2003). Defining the linear portion of a sigmoid-shaped curve: Bend points. Pharmaceutical Statistics - PHARM STAT. 2. 167-174. 10.1002/pst.62. See the pdf in the references folder in this repo, or https://www.researchgate.net/publication/246918700_Defining_the_linear_portion_of_a_sigmoid-shaped_curve_Bend_points


        """
        K = 4.680498579
        A, B, C, D = self.get_params().values()
        y_bend_lower = (A - D) / (1 + 1 / K) + D  # type: ignore
        y_bend_upper = (A - D) / (1 + K) + D  # type: ignore
        return y_bend_lower, y_bend_upper

    def get_params(self, deep=False):
        """
        Get the parameters of the 4PL model, optionally including the estimated LODs.

        Parameters
        ----------
        deep: bool, optional 
            If True, return the LODs as well. Defaults to False.

        Returns
        -------
        dict:
            Dictionary containing the parameters of the 4PL model. If `deep` is True, the dictionary will also contain the estimated LODs.
        """
        self._check_fit()
        if deep:
            return {
                "A": self.A_,
                "B": self.B_,
                "C": self.C_,
                "D": self.D_,
                "LLOD": self.LLOD_,
                "ULOD": self.ULOD_,
                "ULOD_y": self.ULOD_y_,
                "LLOD_y": self.LLOD_y_,
            }
        else:
            return {
                "A": self.A_,
                "B": self.B_,
                "C": self.C_,
                "D": self.D_,
            }

    @staticmethod
    def _four_param_logistic(x, A, B, C, D):
        """4 Parameter Logistic (4PL) model."""
        # For addressing fractional powers of negative numbers
        # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
        z = np.sign(x / C) * np.abs(x / C) ** B

        return ((A - D) / (1.0 + z)) + D

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
        # remove zeros from x_data
        x_indexed_y_data = x_indexed_y_data[x_indexed_y_data.index > 0]
        x_min = np.min(x_indexed_y_data.index)
        x_max = np.max(x_indexed_y_data.index)
        bottom_std_dev = x_indexed_y_data.loc[x_min, "y"].std()
        top_std_dev = x_indexed_y_data.loc[x_max, "y"].std()

        # Calculate LLOD and ULOD of RESPONSE SIGNAL
        llod = self.predict(x_min) + (lower_std_dev_multiplier * bottom_std_dev)
        ulod = self.predict(x_max) - (upper_std_dev_multiplier * top_std_dev)

        # Calculate the limits of detection for CONCENTRATION
        llod_x = self.predict_inverse(llod)
        ulod_x = self.predict_inverse(ulod)
        return llod_x, ulod_x, llod, ulod

    @staticmethod
    def _tangent_line_at_midpoint(x, A, B, C, D):
        """Line with slope = (Derivative of the 4PL curve evaluated at C) and passing through the point C."""
        return -(A - D) * B / (4 * C) * (x - C) + (A + D) / 2

    def tangent_line_at_midpoint(self, x):
        """Return y value of the tangent line at the inflection point (C) of the 4PL curve.

        This is an alternate way to define the linear range of the curve, that is, the part of the curve where it is approximately linear.
        """
        self._check_fit()
        return self._tangent_line_at_midpoint(x, self.A_, self.B_, self.C_, self.D_)

    @staticmethod
    def _tangent_line_at_arbitrary_point(x, g, A, B, C, D):
        """Return y value of line with slope = (Derivative of the 4PL curve evaluated at g) and passing through the point (g, f(g))."""
        derivative_at_g = (
            -1 * (A - D) * B * (g / C) ** (B - 1) / (C * (1 + (g / C) ** B) ** 2)
        )
        print(derivative_at_g)
        line_with_slope_at_g = derivative_at_g * (
            x - g
        ) + FourPLLogistic._four_param_logistic(g, A, B, C, D)
        return line_with_slope_at_g

    def tangent_line_at_arbitrary_point(self, x, g):
        """Return the f(x) where f is the line tangent to the 4PL curve at the point g."""
        self._check_fit()
        return FourPLLogistic._tangent_line_at_arbitrary_point(
            x, g, self.A_, self.B_, self.C_, self.D_
        )

    def fit(self, x_data, y_data, weight_func=None, LOD_func=None, **kwargs):
        """
        Fit the 4 Parameter Logistic (4PL) model.

        x_data: x data points
        y_data: y data points
        weight_func: optional Function that calculates weights from y_data. This is
            passed into the `curve_fit` function where the function minimized is `sum
            ((r / weight_func(y_data)) ** 2)` where r is the residuals.
            Thus for a typical 1/y^2 weighting, `weight_func` should be `lambda
            y_data: y_data`
        """
        x_data = np.float64(x_data)
        y_data = np.float64(y_data)
        df_data = pd.DataFrame({"x": x_data, "y": y_data})
        df_data.sort_values(by="x", inplace=True)

        if LOD_func is None:
            # default LOD_func is to use replicate variance
            LOD_func = self._calculate_lod_replicate_variance

        absolute_sigma = False
        weights = None
        if weight_func is not None:
            weights = weight_func(y_data)
            absolute_sigma = True

        # Initial guess for the parameters
        self.guess_A_ = np.min(y_data)  # type: ignore
        if self.slope_direction_positive is not None:
            self.guess_B_ = 1.0 if self.slope_direction_positive else -1.0
        else:
            # type: ignore
            self.guess_B_ = (
                1.0
                if np.mean(
                    df_data.iloc[: np.minimum(self.slope_guess_num_points_to_use, len(df_data))][  # type: ignore
                        "y"
                    ]
                )
                < np.mean(
                    df_data.iloc[-np.minimum(self.slope_guess_num_points_to_use, len(df_data)) :][  # type: ignore
                        "y"
                    ]
                )
                else -1.0
            )
        self.guess_C_ = np.mean(x_data)  # type: ignore
        self.guess_D_ = np.max(y_data)  # type: ignore
        initial_guess = [self.guess_A_, self.guess_B_, self.guess_C_, self.guess_D_]

        curve_fit_kwargs = {
            "f": self._four_param_logistic,
            "xdata": x_data,
            "ydata": y_data,
            "p0": initial_guess,
            "maxfev": 10000,
            "sigma": weights,
            "absolute_sigma": absolute_sigma,
        }

        # overwrite parameters with any kwargs passed in
        for k, v in kwargs.items():
            curve_fit_kwargs[k] = v

        # Perform the curve fit
        params, cov = curve_fit(**curve_fit_kwargs)
        self.A_, self.B_, self.C_, self.D_ = params
        self.cov_ = cov
        self.LLOD_, self.ULOD_, self.LLOD_y_, self.ULOD_y_ = LOD_func(x_data, y_data)
        return self

    @staticmethod
    def jacobian(x_data, A, B, C, D):
        """Jacobian matrix of the 4PL function with respect to A, B, C, D."""
        z = (x_data / C) ** B

        partial_A = 1.0 / (1.0 + z)
        partial_B = -(z * (A - D) * np.log(np.maximum(x_data / C, np.finfo(float).eps))) / (  # type: ignore
            (1.0 + z) ** 2
        )
        partial_C = (B * z * (A - D)) / (C * (1.0 + z) ** 2)
        partial_D = 1.0 - 1.0 / (1.0 + z)

        # Jacobian matrix
        J = np.array([partial_A, partial_B, partial_C, partial_D]).T
        return J

    def predict_confidence_band(self, x_data):
        """
        Predict confidence bands of data points.

        See:
            https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_graphing_confidence_and_predic.htm
            https://www.graphpad.com/guides/prism/latest/curve-fitting/reg_how_confidence_and_prediction_.htm
            https://stats.stackexchange.com/questions/15423/how-to-compute-prediction-bands-for-non-linear-regression

        """
        if self.cov_ is None:
            raise Exception(
                "Covariance matrix is not available. Please call 'fit' with appropriate data."
            )
        J = self.jacobian(x_data, self.A_, self.B_, self.C_, self.D_)
        pred_var = np.sum((J @ self.cov_) * J, axis=1)

        return np.sqrt(pred_var)

    def predict_prediction_band(self, x_data, y_data):
        """Predict prediction bands of data points.

        TODO: still need to double-check the math here.
        """
        ss = (y_data - self.predict(x_data)) ** 2
        df = len(x_data) - 4  # 4 parameters

        return np.sqrt(self.predict_confidence_band(x_data) ** 2 * ss / df)

    def predict_inverse(
        self, y: Union[float, int, np.ndarray, Iterable[float]], enforce_limits=True
    ):
        """Inverse 4 Parameter Logistic (4PL) model.

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
        self._check_fit()
        if isinstance(y, list):
            y = np.array(y, dtype=float)

        z = ((self.A_ - self.D_) / (y - self.D_)) - 1  # type: ignore

        # For addressing fractional powers of negative numbers, np.sign(z) * np.abs(z) used rather than z
        # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
        x = self.C_ * (np.sign(z) * np.abs(z) ** (1 / self.B_))  # type: ignore
        if enforce_limits:
            if isinstance(y, (np.ndarray, pd.Series)):
                x[y > self.D_] = np.nan  # type: ignore
                x[y < self.A_] = 0  # type: ignore
            elif isinstance(y, (int, float)):
                if y > self.D_:  # type: ignore
                    return np.nan
                elif y < self.A_:  # type: ignore
                    return 0
        return x

    def predict(self, x_data):
        """Predict y-values using the 4PL model.

        Parameters
        ----------
            x_data (iterable): x data points

        Returns
        -------
            iterable: y data points
        """
        self._check_fit()
        return self._four_param_logistic(x_data, self.A_, self.B_, self.C_, self.D_)

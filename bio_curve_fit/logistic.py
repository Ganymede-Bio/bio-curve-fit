from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin

from .base import BaseStandardCurve


class FourPLLogistic(BaseEstimator, RegressorMixin, BaseStandardCurve):
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

    def check_fit(self):
        if self.A_ is None or self.B_ is None or self.C_ is None or self.D_ is None:
            raise Exception(
                "Model is not fit yet. Please call 'fit' with appropriate data"
                " or initialize the model object with non-null parameters."
            )

    def get_params(self, deep=False):
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
    def four_param_logistic(x, A, B, C, D):
        """4 Parameter Logistic (4PL) model."""

        # For addressing fractional powers of negative numbers
        # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
        z = np.sign(x / C) * np.abs(x / C) ** B

        return ((A - D) / (1.0 + z)) + D

    @staticmethod
    def inverse_variance_weight_function(y_data):
        """
        Function for weighting residuals by 1/y^2 in `scipy.optimize.curve_fit`.
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
        """
        Calculate the Lower and Upper Limits of Detection (LLOD and ULOD) using variance
        of replicate max and min concentration standards. It ignore zero concentration
        standards. If there are no replicates, the standard deviation zero
        Possible TODO: sometimes a minimum variance is used in other software.

        In the notation below we assume the response signal is the Y-axis and the
        concentration is the X-axis.

        Example: Two replicates of the lowest concentration standard (conc=1.0 pg/ml)
        have standard deviation of 100 across their responses. LLOD will be `model.predict
        (1.0) + 100 * 2.5` where 2.5 is the `lower_std_dev_multiplier` parameter.

        :param bottom_std_dev: Standard deviation at the bottom calibration point.
        :param top_std_dev: Standard deviation at the top calibration point.
        :param std_dev_multiplier: Multiplier for the standard deviations (default 2.5).
        :return: Pair of tuples containing the LLOD and ULOD, and the corresponding x-values.
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
            "f": self.four_param_logistic,
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
        """
        Jacobian matrix of the 4PL function with respect to A, B, C, D.
        """
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
        TODO: still need to double-check the math here.

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
        """
        Predict prediction bands of data points.
        TODO: still need to double-check the math here.
        """
        ss = (y_data - self.predict(x_data)) ** 2
        df = len(x_data) - 4  # 4 parameters

        return np.sqrt(self.predict_confidence_band(x_data) ** 2 * ss / df)

    def predict_inverse(self, y):
        # TODO: could merge this with `predict` and make it a parameter
        """Inverse 4 Parameter Logistic (4PL) model.

        Used for calculating the x-value for a given y-value.
        Usually, standard curves are fitted using concentration as x-values and response as
        y-values, so that variance in response is modeled for a given known concentration.
        But for samples of unknown concentration, we want to get the concentration as given
        response, which is what this function does.

        """
        self.check_fit()
        z = ((self.A_ - self.D_) / (y - self.D_)) - 1  # type: ignore

        # For addressing fractional powers of negative numbers, np.sign(z) * np.abs(z) used rather than z
        # https://stackoverflow.com/questions/45384602/numpy-runtimewarning-invalid-value-encountered-in-power
        return self.C_ * (np.sign(z) * np.abs(z) ** (1 / self.B_))  # type: ignore

    def predict(self, x_data):
        self.check_fit()
        return self.four_param_logistic(x_data, self.A_, self.B_, self.C_, self.D_)

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class FourPLLogistic(BaseEstimator, RegressorMixin):
    def __init__(self, A=None, B=None, C=None, D=None):
        # A is the minimum asymptote
        self.A_ = A
        # B is the Hill's slope
        self.B_ = B
        # C is the inflection point (EC50)
        self.C_ = C
        # D is the maximum asymptote
        self.D_ = D
        self.cov_ = None

    def get_params(self):
        return self.A_, self.B_, self.C_, self.D_

    @staticmethod
    def four_param_logistic(x, A, B, C, D):
        """4 Parameter Logistic (4PL) model."""
        return ((A - D) / (1.0 + ((x / C) ** B))) + D

    @staticmethod
    def inverse_variance_weight_function(y_data):
        """
        Function for weighting residuals by 1/y^2 in `scipy.optimize.curve_fit`.
        """
        # To avoid division by zero, add a small constant to y_data.
        return y_data + np.finfo(float).eps

    def fit(self, x_data, y_data, weight_func=None):
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
        absolute_sigma = False
        weights = None
        if weight_func is not None:
            weights = weight_func(y_data)
            absolute_sigma = True
        # Initial guess for the parameters

        # initial_guess = [546.75, 1, 652.6776843228683, 221115.26]
        initial_guess = [min(y_data), 1, np.mean(x_data), max(y_data)]
        print(initial_guess)

        # Perform the curve fit
        params, cov = curve_fit(
            self.four_param_logistic,
            x_data,
            y_data,
            p0=initial_guess,
            maxfev=10000,
            sigma=weights,
            absolute_sigma=absolute_sigma,
        )
        self.A_, self.B_, self.C_, self.D_ = params
        self.cov_ = cov
        return self

    def calculate_lod(
        self,
        x_data,
        y_data,
        lower_std_dev_multiplier: float = 2.5,
        upper_std_dev_multiplier: float = 0.0,
    ):
        """
        Calculate the Lower and Upper Limits of Detection (LLOD and ULOD).

        :param bottom_std_dev: Standard deviation at the bottom calibration point.
        :param top_std_dev: Standard deviation at the top calibration point.
        :param std_dev_multiplier: Multiplier for the standard deviations (default 2.5).
        :return: A tuple (LLOD, ULOD).
        """

        x_min = np.min(x_data)
        x_max = np.max(x_data)
        x_indexed_y_data = pd.DataFrame({"x": x_data, "y": y_data}).set_index("x")
        bottom_std_dev = x_indexed_y_data.loc[x_min, "y"].std()  # type: ignore
        top_std_dev = x_indexed_y_data.loc[x_max, "y"].std()  # type: ignore

        # Calculate LLOD
        print(self.predict(x_min), bottom_std_dev, bottom_std_dev * lower_std_dev_multiplier)
        llod = self.predict(x_min) + (lower_std_dev_multiplier * bottom_std_dev)
        # Calculate ULOD
        ulod = self.predict(x_max) - (upper_std_dev_multiplier * top_std_dev)

        return llod, ulod

    def predict_inverse(self, y):
        """Inverse 4 Parameter Logistic (4PL) model.

        Used for calculating the x-value for a given y-value.
        Usually, standard curves are fitted using concentration as x-values and response as y-values, so that variance in response is modeled for a given known concentration. But for samples of unknown concentration, we want to get the concentration as given response, which is what this function does.

        """
        return self.C_ * (((self.A_ - self.D_) / (y - self.D_)) - 1) ** (1 / self.B_)  # type: ignore

    def predict(self, x_data):
        if self.A_ is None or self.B_ is None or self.C_ is None or self.D_ is None:
            raise Exception("Model is not fitted yet. Please call 'fit' with appropriate data.")

        return self.four_param_logistic(x_data, self.A_, self.B_, self.C_, self.D_)

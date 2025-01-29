"""Interfaces for standard curve models."""

from abc import ABC, abstractmethod
from typing import Optional

from sklearn.base import BaseEstimator  # type: ignore


class BaseStandardCurve(ABC, BaseEstimator):
    """Interface for standard curve models."""

    # Upper and lower Limits of Detection ("LODs")
    # Estimated Limits of Detection for concentration
    LLOD: Optional[float]
    ULOD: Optional[float]
    # Estimated Limits of Detection for response signal
    LLOD_y_: Optional[float]
    ULOD_y_: Optional[float]

    @abstractmethod
    def predict(self, x):
        """To be overwritten by derived classes."""
        return None

    @abstractmethod
    def predict_inverse(self, y):
        """
        Predict the inverse (x-value for a given y-value).

        Often standard curves are fit to data where the x-values are known
        concentrations an y-values are measured responses. Later, we usually wish
        to predict the concentration for a given response (i.e. x given y)
        """
        pass

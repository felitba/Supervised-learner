import numpy as np

from src.activation.activation import ActivationFunction, Array


class ReLUActivation(ActivationFunction):
    """θ(h) = max(0, h),  θ'(h) = 1 if h > 0 else 0"""

    def compute(self, h: Array) -> Array:
        """θ(h) = max(0, h)"""
        return np.maximum(0, h)

    def derivative(self, h: Array) -> Array:
        """θ'(h) = 1 if h > 0 else 0"""
        return (h > 0).astype(float)

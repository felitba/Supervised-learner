import numpy as np
from src.activation.activation import ActivationFunction, Array


class LogisticActivation(ActivationFunction):
    """θ(h) = 1 / (1 + e^(-β·h)),  θ'(h) = β·θ(h)·(1 - θ(h))"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta

    def compute(self, h: Array) -> Array:
        """θ(h) = 1 / (1 + e^(-β·h))"""
        return 1.0 / (1.0 + np.exp(-self.beta * h))

    def derivative(self, h: Array) -> Array:
        """θ'(h) = β·θ(h)·(1 - θ(h))"""
        o = self.compute(h)
        return self.beta * o * (1.0 - o)
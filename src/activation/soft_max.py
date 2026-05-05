import numpy as np

from activation.activation import ActivationFunction, Array


class SoftMaxActivation(ActivationFunction):

    def compute(self, h: Array) -> Array:
        # prevents overflow and does not alternate the result
        h_shift = h - np.max(h)
        exp_h = np.exp(h_shift)
        return exp_h / np.sum(exp_h)

    def derivative(self, h: Array) -> Array:
        """
        Jacobian of softmax:
        J = diag(s) - s s^T
        """
        s = self.compute(h)
        return np.diag(s) - np.outer(s, s)

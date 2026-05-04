import numpy as np
from src.optimizer.optimizer import Optimizer
from src.activation.activation import Array


class MomentumOptimizer(Optimizer):
    """Gradient descent con momentum. v = β·v - η·∂E/∂w,  Δw = v"""

    def __init__(self, learning_rate: float, beta: float = 0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self._velocities: list[tuple[Array, Array]] = []

    def update(self, params, grads):
        # Initialize velocities if first call
        if not self._velocities:
            self._velocities = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]

        updated = []
        new_velocities = []

        for (w, b), (gw, gb), (vel_weights, vel_bias) in zip(params, grads, self._velocities):
            vel_weights = self.beta * vel_weights + (1.0 - self.beta) * gw
            vel_weights = self.beta * vel_weights + gw

            w_new = w - self.learning_rate * vel_weights
            b_new = b - self.learning_rate * vel_bias

            updated.append((w_new, b_new))
            new_velocities.append((vel_weights, vel_bias))

        self._velocities = new_velocities
        return updated

    def reset(self) -> None:
        self._velocities = []

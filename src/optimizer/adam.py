import numpy as np
from src.optimizer.optimizer import Optimizer
from src.activation.activation import Array


class AdamOptimizer(Optimizer):
    """Adam optimizer. Combina momentum y RMSprop con corrección de sesgo.

    m = β₁·m + (1-β₁)·g
    v = β₂·v + (1-β₂)·g²
    m̂ = m / (1 - β₁ᵗ),  v̂ = v / (1 - β₂ᵗ)
    Δw = -η · m̂ / (√v̂ + ε)
    """

    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._t: int = 0
        self._m: list[tuple[Array, Array]] = []
        self._velocities: list[tuple[Array, Array]] = []

    def update(
        self,
        params: list[tuple[Array, Array]],
        grads:  list[tuple[Array, Array]],
    ) -> list[tuple[Array, Array]]:
         if not self._velocities:
            self._velocities = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
         if not self._m:
            self._m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]

         updated = []
         new_velocities = []
         new_m = []
         self._t += 1
         for (w, b), (gw, gb), (vel_weights, vel_bias), (m_weights, m_bias) in zip(params, grads, self._velocities, self._m):
            # Calculate new m

            m_weights = self.beta1 * m_weights + (1 - self.beta1) * gw
            m_bias = self.beta1 * m_bias + (1 - self.beta1) * gb

            # Calculate new velocities
            vel_weights = self.beta2 * vel_weights + (1 - self.beta2) * (gw ** 2)
            vel_bias = self.beta2 * vel_bias + (1 - self.beta2) * (gb **2)


            # Apply Bias correction for m and velocities
            m_weights_adjusted = m_weights / (1 - self.beta1 ** self._t)
            m_bias_adjusted = m_bias / (1 - self.beta1 ** self._t)

            vel_weights_adjusted = vel_weights / (1 - self.beta2 ** self._t)
            vel_bias_adjusted = vel_bias / (1 - self.beta2 ** self._t)

            # Update weights
            w_new = w - self.learning_rate * (m_weights_adjusted / (np.sqrt(vel_weights_adjusted) + self.epsilon))
            bias_new = b - self.learning_rate * (m_bias_adjusted / (np.sqrt(vel_bias_adjusted) + self.epsilon))



            # Store new values for next iteration
            updated.append((w_new, bias_new))
            new_velocities.append((vel_weights, vel_bias))
            new_m.append((m_weights, m_bias))

         self._velocities = new_velocities
         self._m = new_m
         return updated

    def reset(self) -> None:
        self._t = 0
        self._m = []
        self._velocities = []

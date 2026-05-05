import numpy as np

from src.cost.cost import CostFunction
from src.activation.activation import Array


class CategoricalCrossEntropyCost(CostFunction):
    """E(ζ, O) = -1/N Σᵢ Σₖ ζᵢₖ log(Oᵢₖ)"""

    def compute(self, zeta: Array, O: Array) -> float:
        """E = -1/N Σ Σ ζ log(O)"""
        O_clipped = np.clip(O, 1e-12, 1.0 - 1e-12)
        return -np.sum(zeta * np.log(O_clipped))

    def gradient(self, zeta: Array, O: Array) -> Array:
        """∂E/∂O = -ζ / O / N"""
        return -zeta / O

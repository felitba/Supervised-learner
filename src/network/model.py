from abc import ABC, abstractmethod
import numpy as np
from src.activation.activation import Array


class Model(ABC):
    """Cualquier red que pueda entrenarse con Trainer debe implementar esta interfaz."""

    @abstractmethod
    def forward(self, x: Array) -> Array: ...

    @abstractmethod
    def backward(self, grad_output: Array) -> None: ...

    @abstractmethod
    def get_weights(self) -> list[tuple[Array, Array]]: ...

    @abstractmethod
    def get_grads(self) -> list[tuple[Array, Array]]: ...

    @abstractmethod
    def set_weights(self, weights: list[tuple[Array, Array]]) -> None: ...

    def zero_grads(self) -> None: ...

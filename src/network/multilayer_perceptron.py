from src.network.model import Model
from src.network.neuron_layer import NeuronLayer
from src.activation.activation import Array


class MultilayerPerceptron(Model):
    """Lista de NeuronLayer. Implementa feed-forward y backpropagation.

    Un perceptrón simple es un MLP con una sola NeuronLayer de una neurona.
    """

    def __init__(self, layers: list[NeuronLayer]) -> None:
        self.layers = layers

    def forward(self, x: Array) -> Array:
        """Propaga x por todas las capas: x → V¹ → V² → ... → O"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad_output: Array) -> None:
        """Retropropaga δ por todas las capas en orden inverso (regla de la cadena)."""
        delta = grad_output
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def get_weights(self) -> list[tuple[Array, Array]]:
        """Retorna lista de (weights, bias) por capa."""
        return [layer.get_weights() for layer in self.layers]

    def get_grads(self) -> list[tuple[Array, Array]]:
        """Retorna lista de (grad_weights, grad_bias) por capa."""
        return [(layer.grad_weights, layer.grad_bias) for layer in self.layers]

    def zero_grads(self) -> None:
        """Resetea los gradientes acumulados de todas las capas."""
        for layer in self.layers:
            layer.zero_grads()

    def set_weights(self, weights: list[tuple[Array, Array]]) -> None:
        """Asigna pesos a todas las capas."""
        for layer, (w, b) in zip(self.layers, weights):
            layer.set_weights(w, b)

    def clone(self) -> "MultilayerPerceptron":
        """Returns a new MLP with the same architecture but freshly initialised weights."""
        return MultilayerPerceptron([
            NeuronLayer(n_inputs=layer.n_inputs, n_neurons=layer.n_neurons, activation=layer.activation)
            for layer in self.layers
        ])

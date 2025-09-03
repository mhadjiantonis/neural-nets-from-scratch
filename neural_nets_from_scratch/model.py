import numpy

from .layer import DenseLayer


class SequentialModel:
    layers: list[DenseLayer]

    def __init__(self, layers: list[DenseLayer]):
        self.layers: list[DenseLayer] = []
        for layer in layers:
            self.add_layer(layer)

    def add_layer(self, layer: DenseLayer):
        if self.layers:
            if self.layers[-1].output_size != layer.input_size:
                raise ValueError(
                    "Input size of the new layer must match the output size of the previous layer."
                )
        self.layers.append(layer)

    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        activations = input
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def backward(self, external_errors: numpy.ndarray):
        errors = external_errors
        for layer in reversed(self.layers):
            errors = layer.backward(errors)

    def update_weights(self, learning_rate: float):
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def get_weights(self) -> dict[int, dict[str, numpy.ndarray]]:
        weights = {
            i: {"weights": layer.weights, "biases": layer.biases}
            for i, layer in enumerate(self.layers)
        }
        return weights

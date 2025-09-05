import pickle
from typing import Self

import numpy

from .layer import DenseLayer
from .utils import shuffle_data


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

    def forward(
        self, input: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        activations = input
        for layer in self.layers:
            activations = layer.forward(activations)
        return activations

    def backward(self, external_errors: numpy.typing.NDArray[numpy.float64]):
        errors = external_errors
        for layer in reversed(self.layers):
            errors = layer.backward(errors)

    def update_weights(self, learning_rate: float):
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def get_weights(
        self,
    ) -> dict[int, dict[str, numpy.typing.NDArray[numpy.float64]]]:
        weights = {
            i: {"weights": layer.weights, "biases": layer.biases}
            for i, layer in enumerate(self.layers)
        }
        return weights

    def train(
        self,
        X: numpy.typing.NDArray[numpy.float64],
        Y: numpy.typing.NDArray[numpy.float64],
        X_test: numpy.typing.NDArray[numpy.float64],
        Y_test: numpy.typing.NDArray[numpy.float64],
        *,
        learning_rate: float = 0.1,
        batch_size: int = 200,
        num_epochs: int = 5,
    ):
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have the same number of rows")

        sample_size = X.shape[0]

        for epoch in range(num_epochs):
            X_epoch, Y_epoch = shuffle_data(X, Y)
            i = 0
            while i < sample_size:
                # print(f"Training on batch {i // batch_size + 1}")
                X_batch = X_epoch[i : min(sample_size, i + batch_size)]
                Y_batch = Y_epoch[i : min(sample_size, i + batch_size)]
                outpupt = self.forward(X_batch)
                # print(numpy.sum(Y_batch * numpy.log(outpupt)))
                errors = -Y_batch / outpupt
                self.backward(errors.T)
                self.update_weights(learning_rate=learning_rate)
                i += batch_size
            print(
                f"Loss at end of epoch {epoch + 1}: {- numpy.sum(Y_test * numpy.log(self.forward(X_test)))}"
            )

    def save(self, path: str):
        with open(path, "wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as file:
            model = pickle.load(file)
        if not isinstance(model, cls):
            raise TypeError("File does not contain a valid model")
        return model

from abc import ABC, abstractmethod
from typing import Callable

import numpy


class ActivationFunction(ABC):

    @abstractmethod
    def func(self, a: numpy.ndarray) -> numpy.ndarray:
        pass

    @abstractmethod
    def derivative(self, a: numpy.ndarray) -> numpy.ndarray:
        pass


class ReLUActivation(ActivationFunction):

    def func(self, a: numpy.ndarray) -> numpy.ndarray:
        return a.clip(min=0.0)

    def derivative(self, a: numpy.ndarray) -> numpy.ndarray:
        return numpy.where(a > 0.0, 1.0, 0.0)


class SoftMaxActivation(ActivationFunction):

    def func(self, a: numpy.ndarray) -> numpy.ndarray:
        exp = numpy.exp(a)
        return exp / exp.sum()

    def derivative(self, a: numpy.ndarray) -> numpy.ndarray:
        func = self.func(a)
        return func - numpy.pow(func, 2)


class DenseLayer:
    input_size: int
    output_size: int
    activation_function: ActivationFunction
    weights: numpy.ndarray
    biases: numpy.ndarray
    input: numpy.ndarray
    output: numpy.ndarray
    derivative_output: numpy.ndarray

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation_function: ActivationFunction,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.weights = numpy.random.rand(input_size, output_size) - 0.5
        self.biases = numpy.random.rand(output_size) - 0.5

    def forward(self, input: numpy.ndarray) -> numpy.ndarray:
        self.input = input
        linear = input.dot(self.weights) + self.biases
        output = self.activation_function.func(linear)
        self.output = output
        self.derivative_output = self.activation_function.derivative(linear)
        return output

    def backward(self, errors: numpy.ndarray) -> numpy.ndarray:
        self.errors = errors
        upstream_errors = self.weights.dot(errors * self.derivative_output.T)
        return upstream_errors

    def update_weights(self, learning_rate: float):
        self.biases -= learning_rate * numpy.sum(self.errors * self.derivative_output.T)
        self.weights -= numpy.dot(self.input.T, self.errors.T * self.derivative_output)


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


if __name__ == "__main__":

    model = SequentialModel(
        [
            DenseLayer(3, 40, ReLUActivation()),
            DenseLayer(40, 10, ReLUActivation()),
            DenseLayer(10, 5, ReLUActivation()),
            DenseLayer(5, 1, SoftMaxActivation()),
        ]
    )
    dataset_size = 10000

    input = numpy.random.rand(dataset_size, 3) * 1000
    print(model.forward(input))

    external_errors = numpy.random.rand(dataset_size, 1)
    model.backward(external_errors.T)
    # print(model.get_weights())
    model.update_weights(0.1)
    # print(model.get_weights())

import numpy

from .activation_function import ActivationFunction


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

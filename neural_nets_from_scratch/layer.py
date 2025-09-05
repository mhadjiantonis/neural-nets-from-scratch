import numpy

from .activation_function import ActivationFunction


class DenseLayer:
    input_size: int
    output_size: int
    activation_function: ActivationFunction
    weights: numpy.typing.NDArray[numpy.float64]
    biases: numpy.typing.NDArray[numpy.float64]
    input: numpy.typing.NDArray[numpy.float64]
    output: numpy.typing.NDArray[numpy.float64]
    jacobian_output: numpy.typing.NDArray[numpy.float64]

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

    def forward(
        self, input: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        self.input = input
        linear = input.dot(self.weights) + self.biases
        output = self.activation_function.func(linear)
        # print(input.sum(axis=1), linear.sum(axis=1), output.sum(axis=1))
        self.output = output
        self.jacobian_output = self.activation_function.jacobian(linear)
        return output

    def backward(
        self, errors: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        self.errors = errors
        upstream_errors = numpy.einsum(
            "jm,imk,ki->ji",
            self.weights,
            self.jacobian_output,
            errors,
        )
        return upstream_errors

    def update_weights(self, learning_rate: float):
        d_biases = learning_rate * numpy.einsum(
            "ijk,ki", self.jacobian_output, self.errors, optimize=True
        )
        d_weights = learning_rate * numpy.einsum(
            "ij,ikl,li->jk",
            self.input,
            self.jacobian_output,
            self.errors,
            optimize=True,
        )
        self.biases -= d_biases
        self.weights -= d_weights

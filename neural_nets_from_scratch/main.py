import numpy
import pandas

from neural_nets_from_scratch.activation_function import (
    SigmoidActivation,
    SoftMaxActivation,
)
from neural_nets_from_scratch.layer import DenseLayer
from neural_nets_from_scratch.model import SequentialModel

if __name__ == "__main__":

    # Define a model
    model = SequentialModel(
        [
            DenseLayer(784, 13, SigmoidActivation()),
            DenseLayer(13, 5, SigmoidActivation()),
            DenseLayer(5, 10, SoftMaxActivation(temperature=1.0)),
        ]
    )

    X: numpy.typing.NDArray[numpy.float64]
    Y_labels: numpy.typing.NDArray[numpy.int64]
    X_test: numpy.typing.NDArray[numpy.float64]
    Y_test_labels: numpy.typing.NDArray[numpy.int64]

    # Load train and test data
    data = numpy.genfromtxt(
        "mnist-in-csv/mnist_train.csv", dtype=int, delimiter=",", skip_header=1
    )

    data_test = numpy.genfromtxt(
        "mnist-in-csv/mnist_test.csv", dtype=int, delimiter=",", skip_header=1
    )

    # Process data. Enforce 0 < X < 1 and one-hot encode Y.
    X = data[:, 1:].astype(float) / 255
    X_test = data_test[:, 1:].astype(float) / 255

    Y_labels = data[:, 0]
    Y_test_labels = data_test[:, 0]

    Y = numpy.zeros((Y_labels.size, Y_labels.max() + 1), dtype=float)
    Y[numpy.arange(Y_labels.size), Y_labels] = 1

    Y_test = numpy.zeros((Y_test_labels.size, Y_test_labels.max() + 1), dtype=float)
    Y_test[numpy.arange(Y_test_labels.size), Y_test_labels] = 1

    # Train the model
    print(f"Loss pre-training: {- numpy.sum(Y * numpy.log(model.forward(X)))}")

    model.train(X, Y, X_test, Y_test, learning_rate=0.01, batch_size=200, num_epochs=15)

    model.save("model.pickle")

    print(
        pandas.crosstab(
            Y_test_labels,
            model.forward(X_test).argmax(axis=1),
            rownames=["real"],
            colnames=["predicted"],
        )
    )

    model1 = SequentialModel.load("model.pickle")

import numpy

from neural_nets_from_scratch.activation_function import ReLUActivation, SoftMaxActivation
from neural_nets_from_scratch.layer import DenseLayer
from neural_nets_from_scratch.model import SequentialModel

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

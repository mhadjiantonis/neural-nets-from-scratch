import numpy


def shuffle_data(
    X: numpy.typing.NDArray[numpy.float64], Y: numpy.typing.NDArray[numpy.float64]
) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
    data = numpy.concat((X, Y), axis=1)
    numpy.random.shuffle(data)
    X_shuffle = data[:, : X.shape[1]]
    Y_shuffle = data[:, X.shape[1] :]
    return X_shuffle, Y_shuffle

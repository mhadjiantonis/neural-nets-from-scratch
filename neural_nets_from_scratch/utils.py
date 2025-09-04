import numpy

def shuffle_data(
    X: numpy.ndarray, Y: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray]:
    data = numpy.concat((X, Y), axis=1)
    numpy.random.shuffle(data)
    X_shuffle = data[:, : X.shape[1]]
    Y_shuffle = data[:, X.shape[1] :]
    return X_shuffle, Y_shuffle
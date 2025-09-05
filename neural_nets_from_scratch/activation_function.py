from abc import ABC, abstractmethod

import numpy


class ActivationFunction(ABC):

    @abstractmethod
    def func(
        self, a: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        pass

    @abstractmethod
    def jacobian(
        self, a: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        pass


class ReLUActivation(ActivationFunction):

    def func(
        self, a: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        result = a.clip(min=0.0)
        result = result / result.sum(axis=1, keepdims=True)
        return result

    def jacobian(
        self, a: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        func = self.func(a)
        step = numpy.where(a > 0.0, 1.0, 0.0) / a.clip(min=0.0).sum(
            axis=1, keepdims=True
        )
        jacobian = numpy.einsum(
            "ik,jk->ijk", step, numpy.eye(step.shape[1]), optimize=True
        ) - numpy.einsum("ik,ij->ijk", func, step, optimize=True)
        return jacobian


class SoftMaxActivation(ActivationFunction):

    temperature: float

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def func(
        self, a: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        exp = numpy.exp(a / self.temperature)
        return exp / exp.sum(axis=1, keepdims=True)

    def jacobian(
        self, a: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        func = self.func(a)
        jacobian = (
            numpy.einsum("ij,jk->ijk", func, numpy.eye(func.shape[1]), optimize=True)
            - numpy.einsum("ij,ik->ijk", func, func, optimize=True) / self.temperature
        )
        return jacobian


class SigmoidActivation(ActivationFunction):

    def func(
        self, a: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        result = 1.0 / (1.0 + numpy.exp(-a))
        return result

    def jacobian(
        self, a: numpy.typing.NDArray[numpy.float64]
    ) -> numpy.typing.NDArray[numpy.float64]:
        derivatives = numpy.exp(-a) / numpy.pow(1.0 + numpy.exp(-a), 2)
        jacobian = numpy.einsum(
            "ij,jk->ijk", derivatives, numpy.eye(derivatives.shape[1]), optimize=True
        )
        return jacobian

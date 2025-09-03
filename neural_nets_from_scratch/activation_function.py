from abc import ABC, abstractmethod

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

import numpy as np


class ActivationFunction:
    """
    Base class for all activation functions.
    """
    @staticmethod
    def activate(z):
        raise NotImplementedError(
            "activate method should be implemented by subclasses.")

    @staticmethod
    def derivative(z):
        raise NotImplementedError(
            "derivative method should be implemented by subclasses.")


class Sigmoid(ActivationFunction):
    """
    Sigmoid activation function.
    """
    @staticmethod
    def activate(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(z):
        sigmoid = Sigmoid.activate(z)
        return sigmoid * (1 - sigmoid)


class ReLU(ActivationFunction):
    """
    ReLU (Rectified Linear Unit) activation function.
    """
    @staticmethod
    def activate(z):
        return np.maximum(0, z)

    @staticmethod
    def derivative(z):
        return (z > 0).astype(float)


class Softmax(ActivationFunction):
    """
    Softmax activation function.
    """
    @staticmethod
    def activate(z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def derivative(z):
        """
        The derivative of softmax is more complex as it's based on the Jacobian matrix.
        This method is generally not used explicitly; instead, it is calculated during backpropagation.
        """
        softmax = Softmax.activate(z)
        return softmax * (1 - softmax)


class Tanh(ActivationFunction):
    """
    Tanh (hyperbolic tangent) activation function.
    """
    @staticmethod
    def activate(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        tanh_z = np.tanh(z)
        return 1 - tanh_z ** 2

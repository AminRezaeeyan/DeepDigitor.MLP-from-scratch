import cupy as cp


class ActivationFunction:
    @staticmethod
    def activate(z):
        raise NotImplementedError(
            "activate method should be implemented by subclasses.")

    @staticmethod
    def derivative(z):
        raise NotImplementedError(
            "derivative method should be implemented by subclasses.")


class Sigmoid(ActivationFunction):
    @staticmethod
    def activate(z):
        return 1 / (1 + cp.exp(-z))

    @staticmethod
    def derivative(z):
        sigmoid = Sigmoid.activate(z)
        return sigmoid * (1 - sigmoid)


class ReLU(ActivationFunction):
    @staticmethod
    def activate(z):
        return cp.maximum(0, z)

    @staticmethod
    def derivative(z):
        return (z > 0).astype(float)


class Softmax(ActivationFunction):
    @staticmethod
    def activate(z):
        exp_z = cp.exp(z - cp.max(z, axis=-1, keepdims=True))
        return exp_z / cp.sum(exp_z, axis=-1, keepdims=True)

    @staticmethod
    def derivative(z):
        softmax = Softmax.activate(z)
        return softmax * (1 - softmax)


class Tanh(ActivationFunction):
    @staticmethod
    def activate(z):
        return cp.tanh(z)

    @staticmethod
    def derivative(z):
        tanh_z = cp.tanh(z)
        return 1 - tanh_z ** 2

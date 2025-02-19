import numpy as np


class LossFunction:
    """
    Base class for all loss functions.
    """
    @staticmethod
    def compute(y_true, y_pred):
        raise NotImplementedError(
            "compute method should be implemented by subclasses.")

    @staticmethod
    def derivative(y_true, y_pred):
        raise NotImplementedError(
            "derivative method should be implemented by subclasses.")


class MeanSquaredError(LossFunction):
    """
    Mean Squared Error (MSE) loss function.
    """
    @staticmethod
    def compute(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def derivative(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size


class CrossEntropyLoss(LossFunction):
    """
    Cross-entropy loss function, commonly used for classification problems.
    """
    @staticmethod
    def compute(y_true, y_pred):
        epsilon = 1e-12  # To prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def derivative(y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -y_true / y_pred


class HingeLoss(LossFunction):
    """
    Hinge loss, often used for binary classification with Support Vector Machines (SVMs).
    """
    @staticmethod
    def compute(y_true, y_pred):
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        return np.where(y_true * y_pred < 1, -y_true, 0)


class HuberLoss(LossFunction):
    """
    Huber loss, a combination of Mean Squared Error and Mean Absolute Error, which is less sensitive to outliers.
    """
    @staticmethod
    def compute(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    @staticmethod
    def derivative(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        return np.where(is_small_error, -error, -delta * np.sign(error))

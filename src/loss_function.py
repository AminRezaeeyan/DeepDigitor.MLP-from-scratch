import cupy as cp


class LossFunction:
    @staticmethod
    def compute(y_true, y_pred):
        raise NotImplementedError(
            "compute method should be implemented by subclasses.")

    @staticmethod
    def derivative(y_true, y_pred):
        raise NotImplementedError(
            "derivative method should be implemented by subclasses.")


class MeanSquaredError(LossFunction):
    @staticmethod
    def compute(y_true, y_pred):
        return cp.mean((y_true - y_pred) ** 2)

    @staticmethod
    def derivative(y_true, y_pred):
        return -2 * (y_true - y_pred) / y_true.size


class CrossEntropyLoss(LossFunction):
    @staticmethod
    def compute(y_true, y_pred):
        epsilon = 1e-12  # To prevent log(0)
        y_pred = cp.clip(y_pred, epsilon, 1. - epsilon)
        return -cp.sum(y_true * cp.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def derivative(y_true, y_pred):
        epsilon = 1e-12
        y_pred = cp.clip(y_pred, epsilon, 1. - epsilon)
        return (y_pred - y_true)


class HingeLoss(LossFunction):
    @staticmethod
    def compute(y_true, y_pred):
        return cp.mean(cp.maximum(0, 1 - y_true * y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        return cp.where(y_true * y_pred < 1, -y_true, 0)


class HuberLoss(LossFunction):
    @staticmethod
    def compute(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        is_small_error = cp.abs(error) <= delta
        squared_loss = 0.5 * error ** 2
        linear_loss = delta * (cp.abs(error) - 0.5 * delta)
        return cp.mean(cp.where(is_small_error, squared_loss, linear_loss))

    @staticmethod
    def derivative(y_true, y_pred, delta=1.0):
        error = y_true - y_pred
        is_small_error = cp.abs(error) <= delta
        return cp.where(is_small_error, -error, -delta * cp.sign(error))

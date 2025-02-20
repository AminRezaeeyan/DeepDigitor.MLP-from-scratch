import numpy as np


class Optimizer:
    """
    Base class for optimizers.
    """

    def update(self, weights, biases, dW, dB):
        raise NotImplementedError(
            "This method should be implemented by subclasses.")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, biases, dW, dB):
        new_weights = weights - self.learning_rate * dW
        new_biases = biases - self.learning_rate * dB
        return new_weights, new_biases


class Adam(Optimizer):
    """
    Adam Optimizer: Adaptive Moment Estimation.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1  # Momentum decay
        self.beta2 = beta2  # RMSProp decay
        self.epsilon = epsilon  # To avoid division by zero
        self.t = 0  # Time step for bias correction
        self.m_w, self.v_w = 0, 0  # First and second moment for weights
        self.m_b, self.v_b = 0, 0  # First and second moment for biases

    def update(self, weights, biases, dW, dB):
        """
        Update weights and biases using Adam optimization.
        """
        self.t += 1

        # Update biased first moment estimates
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * dB

        # Update biased second moment estimates
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dW ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (dB ** 2)

        # Correct bias for first moment
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)

        # Correct bias for second moment
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        # Update weights and biases
        new_weights = weights - self.learning_rate * \
            m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        new_biases = biases - self.learning_rate * \
            m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

        return new_weights, new_biases

import cupy as cp

class Optimizer:
    def update(self, weights, biases, dW, dB):
        raise NotImplementedError("This method should be implemented by subclasses.")


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, biases, dW, dB):
        new_weights = weights - self.learning_rate * dW
        new_biases = biases - self.learning_rate * dB
        return new_weights, new_biases

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate  # Step size
        self.beta1 = beta1                  # Exponential decay rate for first moment
        self.beta2 = beta2                  # Exponential decay rate for second moment
        self.epsilon = epsilon              # Small constant to prevent division by zero
        self.m_w = 0                        # Initialize first moment for weights
        self.v_w = 0                        # Initialize second moment for weights
        self.m_b = 0                        # Initialize first moment for biases
        self.v_b = 0                        # Initialize second moment for biases
        self.t = 0                          # Initialize timestep

    def update(self, weights, biases, dW, dB):
        self.t += 1  # Increment timestep

        # Update biased first moment estimates
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dW
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * dB

        # Update biased second moment estimates
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dW ** 2)
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (dB ** 2)

        # Compute bias-corrected first moment estimates
        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)

        # Compute bias-corrected second moment estimates
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        # Update weights and biases
        new_weights = weights - self.learning_rate * m_w_hat / ((v_w_hat ** 0.5) + self.epsilon)
        new_biases = biases - self.learning_rate * m_b_hat / ((v_b_hat ** 0.5) + self.epsilon)

        return new_weights, new_biases
import cupy as cp
from activation_function import ActivationFunction
from optimizer import SGD


class Layer:
    def __init__(self, input_size, num_neurons, activation_function, optimizer, dropout_rate=0.0):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate

        self.weights = cp.random.randn(
            input_size, num_neurons) * cp.sqrt(2.0 / input_size)
        self.biases = cp.zeros((1, num_neurons))
        self.dropout_mask = None

    def forward(self, inputs, training=True):
        self.inputs = cp.asarray(inputs)
        self.z = cp.dot(self.inputs, self.weights) + self.biases
        self.a = self.activation_function.activate(self.z)

        # Skip dropout on output layer
        if training and self.dropout_rate > 0.0 and self.a.shape[1] != 10:
            self.dropout_mask = cp.random.binomial(
                1, 1 - self.dropout_rate, size=self.a.shape) / (1 - self.dropout_rate)
            self.a *= self.dropout_mask
        else:
            self.dropout_mask = None

        return self.a

    def backward(self, dA, l1_lambda=0.0, l2_lambda=0.0):
        if self.dropout_mask is not None:
            dA *= self.dropout_mask

        dZ = dA * self.activation_function.derivative(self.z)
        dW = cp.dot(self.inputs.T, dZ) / self.inputs.shape[0]
        dB = cp.sum(dZ, axis=0, keepdims=True) / self.inputs.shape[0]

        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, dW, dB)
        return cp.dot(dZ, self.weights.T)

import numpy as np
from perceptron import Perceptron
from activation_function import ActivationFunction


class Layer:
    def __init__(self, input_size: int, num_neurons: int, activation_function: ActivationFunction):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.weights = np.random.randn(input_size, num_neurons)
        self.biases = np.random.randn(1, num_neurons)

    def forward(self, inputs):
        """
        Forward pass through the layer.

        Args:
            inputs (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Activated output of the layer.
        """
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function.activate(self.z)
        return self.a

    def backward(self, dA, learning_rate):
        """
        Backward pass through the layer.

        Args:
            dA (np.ndarray): Gradient of the loss with respect to the output.
            learning_rate (float): Learning rate for weight updates.
        """
        dZ = dA * self.activation_function.derivative(self.z)
        dW = np.dot(self.inputs.T, dZ) / self.inputs.shape[0]
        dB = np.sum(dZ, axis=0, keepdims=True) / self.inputs.shape[0]

        self.weights -= learning_rate * dW
        self.biases -= learning_rate * dB

        return np.dot(dZ, self.weights.T)

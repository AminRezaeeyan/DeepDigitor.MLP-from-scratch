import numpy as np
from perceptron import Perceptron
from activation_function import ActivationFunction
from optimizer import SGD


class Layer:
    def __init__(self, input_size: int, num_neurons: int, activation_function: ActivationFunction, optimizer=SGD()):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.weights = np.random.randn(input_size, num_neurons)
        self.biases = np.random.randn(1, num_neurons)
        self.optimizer = optimizer

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function.activate(self.z)
        return self.a

    def backward(self, dA):
        dZ = dA * self.activation_function.derivative(self.z)
        dW = np.dot(self.inputs.T, dZ) / self.inputs.shape[0]
        dB = np.sum(dZ, axis=0, keepdims=True) / self.inputs.shape[0]

        # Update weights and biases using the optimizer
        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, dW, dB)

        return np.dot(dZ, self.weights.T)

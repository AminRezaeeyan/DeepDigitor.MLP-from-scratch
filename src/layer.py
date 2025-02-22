import cupy as cp
from activation_function import ActivationFunction
from optimizer import SGD


class Layer:
    def __init__(self, input_size, num_neurons, activation_function, optimizer=SGD()):
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.activation_function = activation_function
        self.optimizer = optimizer

        self.weights = cp.random.randn(
            input_size, num_neurons) * cp.sqrt(2.0 / input_size)
        # Use zeros for better stability
        self.biases = cp.zeros((1, num_neurons))

    def forward(self, inputs):
        self.inputs = cp.asarray(inputs)
        self.z = cp.dot(self.inputs, self.weights) + self.biases
        self.a = self.activation_function.activate(self.z)
        return self.a

    def backward(self, dA):
        dZ = dA * self.activation_function.derivative(self.z)
        dW = cp.dot(self.inputs.T, dZ) / self.inputs.shape[0]
        dB = cp.sum(dZ, axis=0, keepdims=True) / self.inputs.shape[0]

        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases, dW, dB)

        return cp.dot(dZ, self.weights.T)

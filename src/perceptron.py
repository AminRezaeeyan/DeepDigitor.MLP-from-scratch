import numpy as np


class Perceptron:
    """
    A single perceptron representing a basic unit in a neural network.
    """

    def __init__(self, input_size: int):
        """
        Initializes the perceptron with random weights and bias.

        Args:
            input_size (int): Number of input features.
        """
        self.input_size = input_size
        self.weights = self.initialize_weights(input_size)
        self.bias = self.initialize_bias()

    @staticmethod
    def initialize_weights(input_size):
        """
        Initialize weights randomly for the perceptron.

        Args:
            input_size (int): Number of input features.

        Returns:
            list: A list of weights initialized randomly.
        """
        return np.random.randn(input_size)

    @staticmethod
    def initialize_bias():
        """
        Initialize the bias randomly.

        Returns:
            float: A randomly initialized bias.
        """
        return np.random.randn()

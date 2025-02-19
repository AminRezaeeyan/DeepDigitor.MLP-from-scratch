from perceptron import Perceptron


class Layer:
    """
    Represents a dense layer consisting of multiple perceptrons.
    """

    def __init__(self, input_size: int, num_neurons: int):
        """
        Initializes a layer with a specified number of perceptrons.

        Args:
            input_size (int): Number of input features.
            num_neurons (int): Number of perceptrons in the layer.
        """
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.perceptrons = self.create_perceptrons()

    def create_perceptrons(self):
        """
        Creates the perceptrons for the layer.

        Returns:
            list: A list of Perceptron objects.
        """
        return [Perceptron(self.input_size) for _ in range(self.num_neurons)]

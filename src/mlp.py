from layer import Layer


class MLP:
    """
    Represents a Multi-Layer Perceptron (MLP) consisting of multiple layers.
    """

    def __init__(self):
        """
        Initializes an empty list of layers.
        """
        self.layers = []

    def add_layer(self, layer: Layer):
        """
        Adds a layer to the MLP.

        Args:
            layer (Layer): The layer to add to the network.
        """
        self.layers.append(layer)

    def summary(self):
        """
        Prints a summary of the MLP structure.
        """
        print("MLP Structure:")
        for idx, layer in enumerate(self.layers, start=1):
            print(f"Layer {idx}: {layer.num_neurons} neurons, Input Size: {layer.input_size}, Activation: {layer.activation_function.__class__.__name__}")

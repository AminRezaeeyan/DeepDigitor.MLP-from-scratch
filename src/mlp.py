from layer import Layer
from loss_function import LossFunction


class MLP:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def forward(self, X):
        """
        Perform a forward pass through the entire network.

        Args:
            X (np.ndarray): Input data of shape (batch_size, input_size).

        Returns:
            np.ndarray: Final output after passing through all layers.
        """
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, X, y_true, learning_rate, loss_function: LossFunction):
        """
        Perform a backward pass through the network and update weights.

        Args:
            X (np.ndarray): Input data.
            y_true (np.ndarray): True labels.
            learning_rate (float): Learning rate for weight updates.
            loss_function (LossFunction): Loss function to compute gradients.
        """
        y_pred = self.forward(X)
        dA = loss_function.derivative(y_true, y_pred)

        for layer in reversed(self.layers):
            dA = layer.backward(dA, learning_rate)

    def summary(self):
        print("MLP Structure:")
        for idx, layer in enumerate(self.layers, start=1):
            print(f"Layer {idx}: {layer.num_neurons} neurons, Input Size: {layer.input_size}, Activation: {layer.activation_function.__class__.__name__}")

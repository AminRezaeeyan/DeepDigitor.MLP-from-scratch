import cupy as cp
from layer import Layer
from loss_function import CrossEntropyLoss


class MLP:
    def __init__(self, loss_function=CrossEntropyLoss, l1_lambda=0.0, l2_lambda=0.0, dropout_rate=0.0):
        self.layers = []
        self.loss_function = loss_function
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate

    def add_layer(self, input_size, num_neurons, activation_function, optimizer):
        self.layers.append(Layer(input_size, num_neurons,
                           activation_function, optimizer, self.dropout_rate))

    def forward(self, X, training=True):
        output = cp.asarray(X)
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, X, y_true, loss_function):
        y_pred = self.forward(X, training=True)
        dA = loss_function.derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA, self.l1_lambda, self.l2_lambda)
        return dA

    def train(self, X, y, epochs=10, batch_size=32, verbose=True, X_val=None, y_val=None):
        n_samples = X.shape[0]
        X = cp.asarray(X)
        y = cp.asarray(y)

        if X_val is not None and y_val is not None:
            X_val = cp.asarray(X_val)
            y_val = cp.asarray(y_val)

        for epoch in range(1, epochs + 1):
            indices = cp.arange(n_samples)
            cp.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                self.backward(X_batch, y_batch, self.loss_function)

            if verbose:
                # No dropout for loss computation
                y_pred = self.forward(X, training=False)
                loss = self.loss_function.compute(y, y_pred)

                l1_loss = 0.0
                l2_loss = 0.0
                for layer in self.layers:
                    l1_loss += cp.sum(cp.abs(layer.weights)) * self.l1_lambda
                    l2_loss += cp.sum(cp.square(layer.weights)
                                      ) * self.l2_lambda
                total_loss = loss + l1_loss + l2_loss

                if X_val is not None and y_val is not None:
                    y_val_pred = self.forward(X_val, training=False)
                    val_loss = self.loss_function.compute(y_val, y_val_pred)
                    # Add regularization to validation loss too
                    val_total_loss = val_loss + l1_loss + l2_loss
                    print(
                        f"Epoch {epoch}/{epochs} - Train Loss: {total_loss:.4f} | Val Loss: {val_total_loss:.4f}")
                else:
                    print(f"Epoch {epoch}/{epochs} - Loss: {total_loss:.4f}")

    def summary(self):
        print("MLP Structure:")
        for idx, layer in enumerate(self.layers, start=1):
            print(f"Layer {idx}: {layer.num_neurons} neurons, Input Size: {layer.input_size}, Activation: {layer.activation_function.__class__.__name__}")

import cupy as cp


class MLP:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = cp.asarray(X)  # Move data to the GPU
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, X, y_true, loss_function):
        y_pred = self.forward(X)
        dA = loss_function.derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def train(self, X, y, loss_function, epochs=10, batch_size=32, verbose=True, X_val=None, y_val=None):
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
                self.backward(X_batch, y_batch, loss_function)

            if verbose:
                y_pred = self.forward(X)
                loss = loss_function.compute(y, y_pred)

                if X_val is not None and y_val is not None:
                    y_val_pred = self.forward(X_val)
                    val_loss = loss_function.compute(y_val, y_val_pred)
                    print(
                        f"Epoch {epoch}/{epochs} - Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")

    def summary(self):
        print("MLP Structure:")
        for idx, layer in enumerate(self.layers, start=1):
            print(f"Layer {idx}: {layer.num_neurons} neurons, Input Size: {layer.input_size}, Activation: {layer.activation_function.__class__.__name__}")

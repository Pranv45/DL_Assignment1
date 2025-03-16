import numpy as np

class FeedforwardNN:
    def __init__(self, input_size, hidden_layers, output_classes=10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_classes = output_classes
        self.weights, self.biases = self._initialize_parameters()

    def _initialize_parameters(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_classes]
        weights = []
        biases = []

        for i in range(1, len(layers)):
            weights.append(np.random.randn(layers[i], layers[i-1]) * 0.01)
            biases.append(np.zeros((layers[i], 1)))

        return weights, biases

    def _relu(self, z):
        return np.maximum(0, z)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def _forward_propagation(self, x):
        activations = [x]
        z_values = []

        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activations[-1]) + b
            z_values.append(z)
            activations.append(self._relu(z))

        # Output layer with softmax activation
        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        z_values.append(z)
        activations.append(self._softmax(z))

        return activations, z_values

    def _compute_loss(self, y_hat, y):
        m = y.shape[1]
        loss = -np.sum(y * np.log(y_hat + 1e-8)) / m
        return loss

    def _backward_propagation(self, activations, z_values, y):
        m = y.shape[1]
        dz = activations[-1] - y
        dw = np.dot(dz, activations[-2].T) / m
        db = np.sum(dz, axis=1, keepdims=True) / m

        dws = [dw]
        dbs = [db]

        for l in range(len(self.hidden_layers), 0, -1):
            da = np.dot(self.weights[l].T, dz)
            dz = da * (z_values[l-1] > 0)
            dw = np.dot(dz, activations[l-1].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            dws.insert(0, dw)
            dbs.insert(0, db)

        return dws, dbs

    def _update_parameters(self, dws, dbs, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dws[i]
            self.biases[i] -= learning_rate * dbs[i]

    def train(self, x_train, y_train, epochs, learning_rate):
        for epoch in range(epochs):
            activations, z_values = self._forward_propagation(x_train)
            loss = self._compute_loss(activations[-1], y_train)
            dws, dbs = self._backward_propagation(activations, z_values, y_train)
            self._update_parameters(dws, dbs, learning_rate)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, x):
        activations, _ = self._forward_propagation(x)
        return np.argmax(activations[-1], axis=0)

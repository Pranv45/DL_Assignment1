import sys
sys.path.append("C:/Users/91831/Desktop/Deep Learning/DL_Assignment1")

import numpy as np
from models.optimizers import SGD, Momentum, Nesterov, RMSProp, Adam, Nadam

class Backpropagation:
    def __init__(self, model, optimizer='sgd', learning_rate=0.01, batch_size=32):
        self.model = model
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = self._initialize_optimizer()

    def _initialize_optimizer(self):
        optimizers = {
            'sgd': SGD,
            'momentum': Momentum,
            'nag': Nesterov,
            'rmsprop': RMSProp,
            'adam': Adam,
            'nadam': Nadam
        }
        if self.optimizer_name not in optimizers:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        return optimizers[self.optimizer_name](self.learning_rate)

    def compute_loss(self, Y_pred, Y_true, loss_function='cross_entropy'):
        if loss_function == 'cross_entropy':
            return -np.sum(Y_true * np.log(Y_pred + 1e-9)) / Y_true.shape[0]
        elif loss_function == 'mean_squared_error':
            return np.mean(np.square(Y_pred - Y_true))
        else:
            raise ValueError("Unsupported loss function")

    def compute_loss_gradient(self, Y_pred, Y_true, loss_function='cross_entropy'):
        m = Y_true.shape[0]
        if loss_function == 'cross_entropy':
            return -(Y_true / (Y_pred + 1e-9)) / m
        elif loss_function == 'mean_squared_error':
            return 2 * (Y_pred - Y_true) / m
        else:
            raise ValueError("Unsupported loss function")

    def backward_pass(self, cache, loss_gradient):
        grads = {}
        dA_prev = loss_gradient

        for i in reversed(range(1, len(self.model.layers))):
            dZ_curr = dA_prev

            if i != len(self.model.layers) - 1:
                activation_derivative = self.model.activation_derivatives.get(
                    self.model.activation_name, lambda x: 1)
                dZ_curr *= activation_derivative(cache[f'Z{i}'])

            dW_curr = cache[f'A{i-1}'].T @ dZ_curr / dZ_curr.shape[0]
            db_curr = np.sum(dZ_curr, axis=0, keepdims=True) / dZ_curr.shape[0]

            if i > 1:
                dA_prev = dZ_curr @ self.model.params[f'W{i}'].T

            grads[f'W{i}'] = dW_curr
            grads[f'b{i}'] = db_curr

        return grads

    def update_parameters(self, grads):
        self.optimizer.update(self.model.params, grads)

    def evaluate(self, X, Y, loss_function):
        Y_pred, _ = self.model.forward(X)
        loss = self.compute_loss(Y_pred, Y, loss_function)
        accuracy = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y, axis=1))
        return loss, accuracy

    def train(self, X_train, y_train_one_hot, epochs=10, validation_data=None, loss_function='cross_entropy'):
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)

            X_train_shuffled = X_train[indices]
            y_train_shuffled_one_hot = y_train_one_hot[indices]

            for batch_start in range(0, X_train.shape[0], self.batch_size):
                batch_end = min(batch_start + self.batch_size, X_train.shape[0])
                X_batch = X_train_shuffled[batch_start:batch_end]
                y_batch_one_hot = y_train_shuffled_one_hot[batch_start:batch_end]

                # Forward pass
                Y_pred_batch, cache_batch = self.model.forward(X_batch)

                # Compute loss and gradients
                loss_gradient_batch = self.compute_loss_gradient(Y_pred_batch, y_batch_one_hot, loss_function)
                grads_batch = self.backward_pass(cache_batch, loss_gradient_batch)

                # Update parameters
                self.update_parameters(grads_batch)

            # Evaluate after each epoch
            train_loss, train_acc = self.evaluate(X_train, y_train_one_hot, loss_function)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            if validation_data:
                X_val, y_val = validation_data
                val_loss, val_acc = self.evaluate(X_val, y_val, loss_function)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        return history

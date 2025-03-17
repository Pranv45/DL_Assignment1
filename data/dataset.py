# Description: This file contains the FashionMNISTLoader class which is used to load the Fashion MNIST dataset.
import numpy as np
from keras.datasets import fashion_mnist

class FashionMNISTLoader:
    def __init__(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data()

    def get_data(self, normalize=True, flatten=True):
        X_train, X_test = self.X_train, self.X_test
        y_train, y_test = self.y_train, self.y_test

        if normalize:
            X_train = X_train.astype(np.float32) / 255.0
            X_test = X_test.astype(np.float32) / 255.0

        if flatten:
            X_train = X_train.reshape(-1, 784)
            X_test = X_test.reshape(-1, 784)

        return (X_train, y_train), (X_test, y_test)

    def get_class_names(self):
        return [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

import sys
sys.path.append("C:/Users/91831/Desktop/Deep Learning/DL_Assignment1")

import numpy as np
from data.dataset import FashionMNISTLoader
from models.feedforward_nn import NeuralNetwork
from models.utils import compute_accuracy, one_hot_encode

def test_feedforward_nn():
    # Load Fashion-MNIST dataset
    loader = FashionMNISTLoader()
    (X_train, y_train), (X_test, y_test) = loader.get_data(normalize=True, flatten=True)

    # Initialize the feedforward neural network
    nn_model = NeuralNetwork(
        input_size=784,        # 28x28 images flattened
        hidden_sizes=[128, 64],  # Two hidden layers: 128 and 64 neurons
        output_size=10,        # 10 classes for Fashion-MNIST
        activation='ReLU',     # Hidden layer activation function
        output_activation='softmax',  # Output layer activation function
        weight_init='Xavier'   # Xavier initialization for weights
    )

    print("Testing Feedforward Neural Network...")

    # Forward pass (no training yet)
    predictions_before_training = nn_model.predict(X_test[:100])  # Predict first 100 samples

    # Compute accuracy before training (random weights)
    accuracy_before_training = compute_accuracy(y_test[:100], predictions_before_training)
    print(f"Accuracy before training (random weights): {accuracy_before_training:.2f}")

    assert accuracy_before_training > 0.0, "Accuracy should be greater than zero"
    assert accuracy_before_training < 0.2, "Accuracy should be low with random weights"

if __name__ == "__main__":
    test_feedforward_nn()

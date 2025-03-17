# Fashion MNIST Neural Network

This project implements a simple feedforward neural network from scratch to classify images from the Fashion MNIST dataset. It includes data loading, forward propagation, backpropagation, and different optimization algorithms.

## Features

- Custom implementation of a feedforward neural network.
- Support for multiple optimization algorithms:
  - Stochastic Gradient Descent (SGD)
  - Momentum
  - Nesterov Accelerated Gradient (NAG)
  - RMSProp
  - Adam
  - Nadam
- Uses NumPy for mathematical computations.
- Integration with Weights & Biases (WandB) for logging and hyperparameter tuning.

## Installation

Ensure you have Python installed along with the required dependencies:

sh
pip install -r requirements.txt


Log in to Weights & Biases:

sh
wandb login <API_KEY>


## Project Structure


DL_ASSIGNMENT1/
├── data/
│   ├── dataset.py         # Contains the FashionMNISTLoader class for loading and preprocessing the dataset
├── models/
│   ├── activations.py     # Implements activation functions (ReLU, Sigmoid, Tanh, etc.)
│   ├── backpropagation.py # Implements backpropagation logic for training the neural network
│   ├── feedforward_nn.py  # Defines the feedforward neural network architecture
│   ├── optimizers.py      # Contains optimization algorithms (SGD, Adam, RMSProp, etc.)
│   ├── utils.py           # Utility functions (e.g., one-hot encoding, accuracy computation)
├── tests/
│   ├── test_feedforward_nn.py  # Unit tests for feedforward neural network implementation
├── wandb/                 # Directory for WandB configuration files
├── train.py               # Script to train the model and perform hyperparameter sweeps using WandB
├── visualize.ipynb        # Jupyter notebook for visualizing training results and metrics
├── README.md              # Project documentation (this file)


## Usage

Run the training script:

sh
python train.py


## Code Overview

### Dataset Loader (data/dataset.py)

FashionMNISTLoader: Loads and preprocesses the Fashion MNIST dataset. Supports normalization and flattening of images.

### Activation Functions (models/activations.py)

Implements common activation functions like Sigmoid, Tanh, ReLU, Identity, and Softmax along with their derivatives.

### Feedforward Neural Network (models/feedforward_nn.py)

NeuralNetwork: A modular implementation of a feedforward neural network with support for multiple hidden layers, custom activation functions, and Xavier weight initialization.

### Backpropagation (models/backpropagation.py)

Implements backpropagation logic for computing gradients and updating parameters using various optimizers.

### Optimizers (models/optimizers.py)

Includes implementations of optimization algorithms such as SGD, Momentum, Nesterov, RMSProp, Adam, and Nadam.

### Utilities (models/utils.py)

Provides utility functions like one-hot encoding and accuracy computation.

### Training Script (train.py)

- Trains the model on the Fashion MNIST dataset.
- Integrates WandB for logging metrics and performing hyperparameter sweeps.

## Hyperparameter Sweep Configuration

The project uses WandB to optimize hyperparameters via sweeps. The sweep configuration includes:

### Parameters:

- *Number of epochs* (epochs)
- *Number of layers* (num_layers)
- *Hidden layer size* (hidden_size)
- *Learning rate* (learning_rate)
- *Optimizer* (optimizer: SGD, Adam, RMSProp)
- *Batch size* (batch_size)
- *Weight decay* (weight_decay)
- *Activation function* (activation: Sigmoid, Tanh, ReLU)
- *Weight initialization method* (weight_init: Random or Xavier)

### Project URL

[Weights & Biases Sweep](https://wandb.ai/me21b118-iit-madras/fashion_mnist_sweep)
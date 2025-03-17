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

## Installation
Ensure you have Python installed along with the required dependencies:

pip install -r requirements.txt

login using 
wandb login <API_KEY>



## Project Structure

- **data/dataset.py**: Loads and preprocesses the Fashion-MNIST dataset.
- **models/activations.py**: Contains implementations for common activation functions like ReLU, Sigmoid, and Tanh.
- **models/backpropagation.py**: Implements the backpropagation algorithm for updating weights during training.
- **models/feedforward_nn.py**: Defines the architecture of the feedforward neural network.
- **models/optimizers.py**: Contains implementations for various optimization algorithms such as SGD, Adam, and RMSProp.
- **models/utils.py**: Utility functions for tasks like one-hot encoding and calculating model accuracy.
- **tests/test_feedforward_nn.py**: Unit tests to validate the functionality of the feedforward neural network.
- **wandb/**: Contains configuration files for tracking experiments using WandB.
- **train.py**: The script that initiates training and hyperparameter tuning via WandB sweeps.
- **visualize.ipynb**: A Jupyter notebook for visualizing the

## Usage
Run the training script:
sh
python train.py


## Code Overview
## Dataset Loader (data/dataset.py)
FashionMNISTLoader: Loads and preprocesses the Fashion MNIST dataset. Supports normalization and flattening of images.

## Activation Functions (models/activations.py)
Implements common activation functions like Sigmoid, Tanh, ReLU, Identity, and Softmax along with their derivatives.

## Feedforward Neural Network (models/feedforward_nn.py)
NeuralNetwork: A modular implementation of a feedforward neural network with support for multiple hidden layers, custom activation functions, and Xavier weight initialization.

## Backpropagation (models/backpropagation.py)
Implements backpropagation logic for computing gradients and updating parameters using various optimizers.

## Optimizers (models/optimizers.py)
Includes implementations of optimization algorithms such as SGD, Momentum, Nesterov, RMSProp, Adam, and Nadam.

## Utilities (models/utils.py)
Provides utility functions like one-hot encoding and accuracy computation.

## Training Script (train.py)
Trains the model on the Fashion MNIST dataset.

Integrates WandB for logging metrics and performing hyperparameter sweeps.

## Hyperparameter Sweep Configuration
The project uses WandB to optimize hyperparameters via sweeps. The sweep configuration includes:

Method: Bayesian optimization (bayes), grid search (grid), or random search (random).

# Parameters:

-Number of epochs (epochs)

-Number of layers (num_layers)

-Hidden layer size (hidden_size)

-Learning rate (learning_rate)

-Optimizer (optimizer: SGD, Adam, RMSProp)

-Batch size (batch_size)

-Weight decay (weight_decay)

-Activation function (activation: Sigmoid, Tanh, ReLU)

Weight initialization method (weight_init: Random or Xavier)
Project URL: https://wandb.ai/me21b118-iit-madras/fashion_mnist_sweep
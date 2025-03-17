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

sh
pip install numpy


## Project Structure

|-- fashion_mnist_loader.py  # Data loading functions
|-- backpropagation.py       # Backpropagation implementation
|-- neural_network.py        # Neural network class
|-- optimizers.py            # Optimizer classes (SGD, Adam, etc.)
|-- main.py                  # Script to train and evaluate the model
|-- README.md                # Project documentation


## Usage
Run the training script:
sh
python train.py


## Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images of size 28x28 pixels, categorized into 10 classes.

## Optimizers
- *SGD:* Basic stochastic gradient descent.
- *Momentum:* Adds velocity to gradient updates.
- *NAG:* Looks ahead at the future gradient direction.
- *RMSProp:* Adapts learning rates using an exponentially weighted average.
- *Adam:* Combines momentum and RMSProp.
- *Nadam:* Adam with Nesterov momentum.

## To-Do
- Improve accuracy with better weight initialization.
- Implement batch normalization.
- Optimize hyperparameters.

Project URL: https://wandb.ai/me21b118-iit-madras/fashion_mnist_sweep
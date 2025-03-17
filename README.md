DA6401_ASSIGNMENT1: Fashion MNIST Classification
Project Description
This project implements a deep learning model to classify images from the Fashion MNIST dataset. It includes custom implementations of feedforward neural networks, backpropagation, activation functions, optimizers, and utilities. The training process is integrated with Weights & Biases (WandB) for hyperparameter sweeps and tracking model performance.

Project Structure
text
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
Requirements
Dependencies
The following Python libraries are required:

numpy

keras

wandb

Install them using:

bash
pip install numpy keras wandb
Setup Instructions
Clone the Repository
bash
git clone <repository-url>
cd DL_ASSIGNMENT1
Install Dependencies
bash
pip install -r requirements.txt
Configure WandB
Sign up at Weights & Biases.

Log in using:

bash
wandb login <API_KEY>
Usage
Training the Model
To train the model with default parameters:

bash
python train.py
Running Hyperparameter Sweeps with WandB
The script supports hyperparameter sweeps using WandB. Modify sweep_config in train.py to define sweep parameters. Initialize and run the sweep agent:

python
sweep_id = wandb.sweep(sweep_config, project="fashion_mnist_sweep")
wandb.agent(sweep_id, function=train_model, count=20)
Testing Feedforward Neural Network
Run unit tests using:

bash
pytest tests/test_feedforward_nn.py
Code Overview
Dataset Loader (data/dataset.py)
FashionMNISTLoader: Loads and preprocesses the Fashion MNIST dataset. Supports normalization and flattening of images.

Activation Functions (models/activations.py)
Implements common activation functions like Sigmoid, Tanh, ReLU, Identity, and Softmax along with their derivatives.

Feedforward Neural Network (models/feedforward_nn.py)
NeuralNetwork: A modular implementation of a feedforward neural network with support for multiple hidden layers, custom activation functions, and Xavier weight initialization.

Backpropagation (models/backpropagation.py)
Implements backpropagation logic for computing gradients and updating parameters using various optimizers.

Optimizers (models/optimizers.py)
Includes implementations of optimization algorithms such as SGD, Momentum, Nesterov, RMSProp, Adam, and Nadam.

Utilities (models/utils.py)
Provides utility functions like one-hot encoding and accuracy computation.

Training Script (train.py)
Trains the model on the Fashion MNIST dataset.

Integrates WandB for logging metrics and performing hyperparameter sweeps.

Hyperparameter Sweep Configuration
The project uses WandB to optimize hyperparameters via sweeps. The sweep configuration includes:

Method: Bayesian optimization (bayes), grid search (grid), or random search (random).

Parameters:

Number of epochs (epochs)

Number of layers (num_layers)

Hidden layer size (hidden_size)

Learning rate (learning_rate)

Optimizer (optimizer: SGD, Adam, RMSProp)

Batch size (batch_size)

Weight decay (weight_decay)

Activation function (activation: Sigmoid, Tanh, ReLU)

Weight initialization method (weight_init: Random or Xavier)

Example configuration:

python
sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [3, 4]},
        'hidden_size': {'values': [64, 128]},
        'learning_rate': {'values': [1e-3]},
        ...
    }
}
Results
Training logs and metrics are synced with WandB. You can view them at:

Project URL: https://wandb.ai/me21b118-iit-madras/fashion_mnist_sweep
 

import wandb
import numpy as np
from models.utils import one_hot_encode
from data.dataset import FashionMNISTLoader
from models.feedforward_nn import NeuralNetwork
from models.backpropogation import Backpropagation

def train_model():
    # Initialize wandb with the sweep hyperparameters
    wandb.init()

    # Get the hyperparameters from the sweep configuration
    config = wandb.config

    # Generate a custom name for the sweep based on the hyperparameters
    sweep_name = f"hl_{config.num_layers}_bs_{config.batch_size}_ac_{config.activation}_lr_{config.learning_rate}_wd_{config.weight_decay}"
    wandb.run.name = sweep_name  # Set the custom sweep name

    # Load and preprocess the dataset
    loader = FashionMNISTLoader()
    (X_train, y_train), (X_test, y_test) = loader.get_data(normalize=True, flatten=True)

    # One-hot encode labels
    y_train_one_hot = one_hot_encode(y_train)
    y_test_one_hot = one_hot_encode(y_test)

    # Split training data into train and validation
    val_split = int(0.9 * X_train.shape[0])
    X_train, X_val = X_train[:val_split], X_train[val_split:]
    y_train_one_hot, y_val_one_hot = y_train_one_hot[:val_split], y_train_one_hot[val_split:]

    # Create the model with the hyperparameters from the sweep
    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=[config.hidden_size] * config.num_layers,
        output_size=10,
        activation=config.activation,
        weight_init=config.weight_init
    )

    # Initialize the backpropagation object
    backprop = Backpropagation(
        model, optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size
    )

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    step = 0
    for epoch in range(config.epochs):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        X_train_shuffled = X_train[indices]
        y_train_shuffled_one_hot = y_train_one_hot[indices]

        for batch_start in range(0, X_train.shape[0], config.batch_size):
            batch_end = min(batch_start + config.batch_size, X_train.shape[0])
            X_batch = X_train_shuffled[batch_start:batch_end]
            y_batch_one_hot = y_train_shuffled_one_hot[batch_start:batch_end]

            Y_pred_batch, cache_batch = model.forward(X_batch)
            loss_gradient = backprop.compute_loss_gradient(Y_pred_batch, y_batch_one_hot, "cross_entropy")
            grads = backprop.backward_pass(cache_batch, loss_gradient)

            backprop.update_parameters(grads)

            batch_loss = backprop.compute_loss(Y_pred_batch, y_batch_one_hot, "cross_entropy")
            batch_acc = np.mean(np.argmax(Y_pred_batch, axis=1) == np.argmax(y_batch_one_hot, axis=1))

            wandb.log({'batch_loss': batch_loss, 'batch_accuracy': batch_acc, 'step': step})
            step += 1

        # Evaluate on training and validation data
        train_loss, train_acc = backprop.evaluate(X_train, y_train_one_hot, "cross_entropy")
        val_loss, val_acc = backprop.evaluate(X_val, y_val_one_hot, "cross_entropy")

        # Save metrics for visualization in wandb
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epoch': epoch + 1
        })

        print(f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    wandb.finish()

# Sweep configuration
sweep_config = {
    'method': 'bayes',  # Or 'grid' or 'random'
    'metric': {'name': 'val_loss', 'goal': 'minimize'},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'adam', 'rmsprop']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'activation': {'values': ['sigmoid', 'tanh', 'ReLU']},
        'weight_init': {'values': ['random', 'Xavier']}
    }
}

# Initialize sweep with the configuration
sweep_id = wandb.sweep(sweep_config, project="fashion_mnist_sweep")

# Start the sweep agent
wandb.agent(sweep_id, function=train_model, count=5)  # You can set count to any number to control how many different runs you want

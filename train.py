import sys
sys.path.append("C:/Users/91831/Desktop/Deep Learning/DL_Assignment1")

import numpy as np
import wandb
import argparse
from models.utils import one_hot_encode
from data.dataset import FashionMNISTLoader
from models.feedforward_nn import NeuralNetwork
from models.backpropogation import Backpropagation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-wp", "--wandb_project", default="my-project-name")
    parser.add_argument("-we", "--wandb_entity", default="me21b118-iit-madras")
    parser.add_argument("-d", "--dataset", choices=["mnist", "fashion_mnist"], default="fashion_mnist")
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-b", "--batch_size", type=int, default=4)
    parser.add_argument("-l", "--loss", choices=["mean_squared_error", "cross_entropy"], default="cross_entropy")
    parser.add_argument("-o", "--optimizer", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], default="adam")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.1)
    parser.add_argument("-m", "--momentum", type=float, default=0.5)
    parser.add_argument("-beta", "--beta", type=float, default=0.5)
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5)
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5)
    parser.add_argument("-eps", "--epsilon", type=float, default=1e-6)
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0)
    parser.add_argument("-w_i", "--weight_init", choices=["random", "Xavier"], default="random")
    parser.add_argument("-nhl", "--num_layers", type=int, default=1)
    parser.add_argument("-sz", "--hidden_size", type=int, default=4)
    parser.add_argument("-a", "--activation", choices=["identity", "sigmoid", "tanh", "ReLU"], default="sigmoid")

    return parser.parse_args()


def train(config):
    wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config)

    data_loader = FashionMNISTLoader()
    (X_train, y_train), (X_val, y_val) = data_loader.get_data()

    y_train_one_hot = one_hot_encode(np.array(y_train), 10)
    y_val_one_hot = one_hot_encode(np.array(y_val), 10)

    model = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_sizes=[config["hidden_size"]] * config["num_layers"],
        output_size=10,
        activation=config["activation"],
        weight_init=config["weight_init"]
    )

    backprop = Backpropagation(
        model, optimizer=config["optimizer"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"]
    )

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    step = 0

    for epoch in range(config["epochs"]):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)

        X_train_shuffled = X_train[indices]
        y_train_shuffled_one_hot = y_train_one_hot[indices]

        for batch_start in range(0, X_train.shape[0], config["batch_size"]):
            batch_end = min(batch_start + config["batch_size"], X_train.shape[0])
            X_batch = X_train_shuffled[batch_start:batch_end]
            y_batch_one_hot = y_train_shuffled_one_hot[batch_start:batch_end]

            Y_pred_batch, cache_batch = model.forward(X_batch)
            loss_gradient = backprop.compute_loss_gradient(Y_pred_batch, y_batch_one_hot, config["loss"])
            grads = backprop.backward_pass(cache_batch, loss_gradient)

            backprop.update_parameters(grads)

            batch_loss = backprop.compute_loss(Y_pred_batch, y_batch_one_hot, config["loss"])
            batch_acc = np.mean(np.argmax(Y_pred_batch, axis=1) == np.argmax(y_batch_one_hot, axis=1))

            wandb.log({'batch_loss': batch_loss, 'batch_accuracy': batch_acc, 'step': step})
            step += 1

        train_loss, train_acc = backprop.evaluate(X_train, y_train_one_hot, config["loss"])
        val_loss, val_acc = backprop.evaluate(X_val, y_val_one_hot, config["loss"])

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

        print(f"Epoch {epoch + 1}/{config['epochs']} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    config = vars(args)  # Convert argparse Namespace to dictionary
    train(config)

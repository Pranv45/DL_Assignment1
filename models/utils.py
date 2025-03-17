# Description: This file contains utility functions that are used in the models.
import numpy as np

def one_hot_encode(y, num_classes=10):
    encoded_y = np.zeros((y.size, num_classes))
    encoded_y[np.arange(y.size), y] = 1
    return encoded_y

def compute_accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

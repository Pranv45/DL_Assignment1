# Description: Feedforward Neural Network class
import sys
sys.path.append("C:/Users/91831/Desktop/Deep Learning/DL_Assignment1")


import numpy as np
from models.activations import sigmoid, tanh, relu, identity, softmax

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size,
                 activation='relu', output_activation='softmax', weight_init='Xavier'):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.activation_name = activation
        self.output_activation_name = output_activation
        
        self.activation_funcs = {
            'sigmoid': sigmoid,
            'tanh': tanh,
            'ReLU': relu,
            'identity': identity,
            'softmax': softmax
        }
        
        self.layers = [input_size] + hidden_sizes + [output_size]
        self.params = self.initialize_weights(weight_init)
    
    def initialize_weights(self, method='Xavier'):
        params = {}
        for i in range(len(self.layers)-1):
            input_dim = self.layers[i]
            output_dim = self.layers[i+1]

            if method == 'Xavier':
                limit = np.sqrt(6 / (input_dim + output_dim))
                params[f'W{i+1}'] = np.random.uniform(-limit, limit, (input_dim, output_dim))
            else:
                params[f'W{i+1}'] = np.random.randn(input_dim, output_dim) * 0.01
            
            params[f'b{i+1}'] = np.zeros((1, output_dim))
        
        return params
    
    def forward(self, X):
        cache = {'A0': X}
        A_prev = X
        
        for i in range(1, len(self.layers)):
            Z_curr = A_prev @ self.params[f'W{i}'] + self.params[f'b{i}']
            
            if i == len(self.layers)-1:
                activation_func = self.activation_funcs[self.output_activation_name]
            else:
                activation_func = self.activation_funcs[self.activation_name]
            
            A_curr = activation_func(Z_curr)
            
            cache[f'Z{i}'] = Z_curr
            cache[f'A{i}'] = A_curr
            
            A_prev = A_curr
        
        return A_curr, cache
    
    def predict(self, X):
        probs, _ = self.forward(X)
        predictions = np.argmax(probs, axis=1)
        return predictions

import sys
sys.path.append("C:/Users/91831/Desktop/Deep Learning/DL_Assignment1")


import numpy as np

class Optimizer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        """
        Update parameters using gradients.
        This is a base class method to be overridden by specific optimizers.
        """
        raise NotImplementedError("This method should be implemented in subclasses.")

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.learning_rate * grads[key]

class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            params[key] += self.velocity[key]

class Nesterov(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            prev_velocity = self.velocity[key]
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            params[key] += -self.momentum * prev_velocity + (1 + self.momentum) * self.velocity[key]

class RMSProp(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-7):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.squared_gradients = {}

    def update(self, params, grads):
        for key in params.keys():
            if key not in self.squared_gradients:
                self.squared_gradients[key] = np.zeros_like(params[key])
            self.squared_gradients[key] = (
                self.beta * self.squared_gradients[key] + (1 - self.beta) * grads[key]**2
            )
            params[key] -= (
                self.learning_rate * grads[key] / (np.sqrt(self.squared_gradients[key]) + self.epsilon)
            )

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment1 = {}
        self.moment2 = {}
        self.timestep = 0

    def update(self, params, grads):
        self.timestep += 1
        for key in params.keys():
            if key not in self.moment1:
                self.moment1[key] = np.zeros_like(params[key])
                self.moment2[key] = np.zeros_like(params[key])

            # Update biased first and second moment estimates
            self.moment1[key] = (
                self.beta1 * self.moment1[key] + (1 - self.beta1) * grads[key]
            )
            self.moment2[key] = (
                self.beta2 * self.moment2[key] + (1 - self.beta2) * grads[key]**2
            )

            # Correct bias in moments
            m_hat = self.moment1[key] / (1 - np.power(self.beta1, self.timestep))
            v_hat = self.moment2[key] / (1 - np.power(self.beta2, self.timestep))

            # Update parameters
            params[key] -= (
                self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            )

class Nadam(Adam):  # Nadam is an extension of Adam with Nesterov momentum
    def update(self, params, grads):
        for key in params.keys():
            if key not in self.moment1:
                # Initialize moments if they don't exist yet
                super().update(params, grads)  # Call Adam's update to initialize moments

            # Apply Nesterov momentum correction to Adam updates
            m_hat_nesterov = (
                (self.beta1 * (self.moment1[key])) + ((1 - 2 * (self.beta1)) / 2) * grads[key])

import numpy as np


class Activation:
    def __init__(self, function, derivative):
        """Works mostly as a pair of function and its derivative, it simply looks better this way."""
        self.function = function
        self.derivative = derivative

    def get_function(self):
        """Used to avoid ambiguous constructions like activation.function(func_params)."""
        return self.function

    def get_derivative(self):
        """Used to avoid ambiguous constructions like activation.derivative(deriv_params)."""
        return self.derivative


linear = Activation(
    lambda x, layer: x,
    lambda x, layer: 1
)
ReLU = Activation(
    lambda x, layer: max(x, 0),
    lambda x, layer: 1 if x > 0 else 0
)
sigmoid = Activation(
    lambda x, layer: (np.exp(-x) + 1) ** (-1),
    lambda x, layer: (np.exp(-x) + 1) ** (-1) * (1 - (np.exp(-x) + 1) ** (-1))
)
tanh = Activation(
    lambda x, layer: 1 - 2 / (np.exp(2 * x) + 1),
    lambda x, layer: 4 * (1 / (np.exp(2 * x) + 1) - 1 / (np.exp(2 * x) + 1) ** 2)
)
softmax = Activation(
    lambda x, layer: np.exp(x) / np.exp(layer).sum(0),
    lambda x, layer: np.exp(x) / np.exp(layer).sum(0) * (1 - np.exp(x) / np.exp(layer).sum(0))
)

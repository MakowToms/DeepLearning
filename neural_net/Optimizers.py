import numpy as np


class Optimizer:
    def __init__(self, optimize_func):
        """Replaces built-in switch with limited number of choices with this interface."""
        self.__optimize__ = optimize_func
        self.params = {}

    def set_params(self, params):
        """Allows setting parameters in form of a dictionary."""
        self.params = params
        return self

    def update_params(self, params):
        """Allows updating parameters in the dictionary, keeping previously set parameters."""
        self.params.update(params)
        return self

    def optimize(self, layer):
        """Used to avoid ambiguous constructions."""
        self.__optimize__(layer, self.params)


def __no_optimizer_func__(layer, params):
    layer.weights_diff = layer.local_gradient.dot(np.transpose(layer.input.values))
    layer.biases_diff = layer.local_gradient.dot(np.ones((layer.input.values.shape[1], 1)))


def __momentum_func__(layer, params):
    if not params.__contains__("coef"):
        raise Exception("Momentum optimizer needs to have \"coef\" parameter! Use `set_params` method.")
    layer.weights_momentum = layer.local_gradient.dot(np.transpose(layer.input.values)) - layer.weights_momentum * params["coef"]
    layer.biases_momentum = layer.local_gradient.dot(np.ones((layer.input.values.shape[1], 1))) - layer.biases_momentum * params["coef"]
    layer.weights_diff = layer.weights_momentum
    layer.biases_diff = layer.biases_momentum


def __rmsprop_func__(layer, params):
    if not params.__contains__("coef"):
        raise Exception("RMSProp optimizer needs to have \"coef\" parameter! Use `set_params` method.")
    layer.weights_diff = layer.local_gradient.dot(np.transpose(layer.input.values))
    layer.biases_diff = layer.local_gradient.dot(np.ones((layer.input.values.shape[1], 1)))
    layer.weights_momentum = layer.weights_momentum * params["coef"] + layer.weights_diff**2 * (1 - params["coef"])
    layer.biases_momentum = layer.biases_momentum * params["coef"] + layer.biases_diff**2 * (1 - params["coef"])
    layer.weights_diff = layer.weights_diff / (layer.weights_momentum**0.5)
    layer.biases_diff = layer.biases_diff / (layer.biases_momentum**0.5)


no_optimizer = Optimizer(__no_optimizer_func__)
# Optimizers below are supposed to have parameter called "coef"
momentum = Optimizer(__momentum_func__)
RMSProp = Optimizer(__rmsprop_func__)

import numpy as np


class Regularization:
    def __init__(self, cost_func, weight_impact):
        self.cost_func = cost_func
        self.weight_impact = weight_impact
        self.params = {}

    def set_params(self, params):
        """Allows setting parameters in form of a dictionary."""
        self.params = params
        return self

    def update_params(self, params):
        """Allows updating parameters in the dictionary, keeping previously set parameters."""
        self.params.update(params)
        return self

    def compute_loss(self, layer):
        """Returns additional cost (loss) associated with layer parameters, usually weights."""
        return self.cost_func(self, layer)

    def update_weights(self, layer):
        """Returns array of values which should be subtracted from weights."""
        return self.weight_impact(self, layer)


no_regularization = Regularization(
    lambda rglr, layer: 0,
    # it's numpy, so simple integer should work
    lambda rglr, layer: 0
)
L1_regularization = Regularization(
    lambda rglr, layer: np.abs(layer.weights).sum().sum() * rglr.params["coef"] / (2 * layer.size),
    # notice np.sign(layer.weights)
    # it's because derivative of |x| is either -1 or 1
    lambda rglr, layer: np.sign(layer.weights) * rglr.params["coef"] / layer.size
)
L2_regularization = Regularization(
    lambda rglr, layer: np.square(layer.weights).sum().sum() * rglr.params["coef"] / (2 * layer.size),
    lambda rglr, layer: layer.weights * rglr.params["coef"] / layer.size
)

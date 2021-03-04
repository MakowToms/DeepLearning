import numpy as np


class Loss:
    def __init__(self, function, derivative):
        self.function = function
        self.derivative = derivative

    def compute_loss(self, y_hat, y):
        """Returns additional cost (loss) associated with layer parameters, usually weights."""
        return self.function(y_hat, y)

    def compute_derivative(self, y_hat, y):
        """Returns derivative of loss."""
        return self.derivative(y_hat, y)


MSE = Loss(
    lambda y_hat, y: np.mean(np.transpose(y_hat - y) ** 2) / 2,
    lambda y_hat, y: y_hat - y
)
MAE = Loss(
    lambda y_hat, y: np.mean(np.transpose(np.abs(y_hat - y))),
    lambda y_hat, y: np.sign(y_hat - y)
)

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


def _log_loss_derivative(y_hat, y):
    losses = np.empty(y.shape)
    for row in range(y.shape[0]):
        losses[row, y[row, :] == 0] = 1 - y_hat[row, y[row, :] == 0]
        losses[row, y[row, :] == 1] = - y_hat[row, y[row, :] == 1]
    return losses


LogLoss = Loss(
    lambda y_hat, y: -np.sum(np.transpose(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))),
    _log_loss_derivative
)

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
hinge = Loss(
    lambda y_hat, y: np.sum(np.transpose(np.maximum(np.equal(y, 0) * (y_hat - y), 0) - np.maximum(np.equal(y, 1) * (y_hat - y), 0))),
    lambda y_hat, y: np.array(np.equal(y, 0) * np.greater(y_hat, 0), dtype=np.int32) - np.equal(y, 1) * np.less(y_hat, 1)
)


def _log_loss_derivative(y_hat, y):
    losses = np.empty(y.shape)
    for row in range(y.shape[0]):
        losses[row, y[row, :] == 0] = 1 - y_hat[row, y[row, :] == 0]
        losses[row, y[row, :] == 1] = - y_hat[row, y[row, :] == 1]
    return losses


LogLoss = Loss(
    lambda y_hat, y: -np.sum(np.transpose(y*np.log(np.maximum(y_hat, np.ones(y_hat.shape)/10**6)) + (1-y)*np.log(np.maximum(1-y_hat, np.ones(y_hat.shape)/10**6)))),
    _log_loss_derivative
)


# it's not loss, but it is useful in metric in classification problem
def accuracy(y_hat, y, threshold=0.5):
    n_observations = y.shape[0]
    proper_classified = np.sum(np.logical_and(y_hat >= threshold, y == 1))
    proper_classified += np.sum(np.logical_and(y_hat < threshold, y == 0))
    return proper_classified / n_observations

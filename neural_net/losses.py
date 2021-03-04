import numpy as np


class Loss:
    def __init__(self, func):
        self.func = func

    def compute_loss(self, y_hat, y):
        """Returns additional cost (loss) associated with layer parameters, usually weights."""
        return self.func(y_hat, y)


MSE = Loss(lambda y_hat, y: np.mean(np.transpose(y_hat - y) ** 2) / 2)

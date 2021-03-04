import numpy as np
import math


class Budget:
    def __init__(self):
        """Takes care of tracking whether training should be stopped."""
        self.epoch = 0
        self.max_epoch = np.inf
        # will stop when last_loss / min(losses) > detection_coef
        self.detection_coef = np.inf
        # limits
        self.epoch_limit = False
        self.detection_limit = False

    def set_epoch_limit(self, max_epoch):
        self.epoch_limit = True
        self.max_epoch = max_epoch
        return self

    def set_detection_limit(self, coef):
        self.detection_limit = True
        self.detection_coef = coef
        return self

    def finished(self, loss_history):
        """Returns True wherever one of stop conditions is fulfilled."""
        if self.epoch_limit:
            if self.epoch >= self.max_epoch:
                return True
        # in case loss_history was just initialized and has no history
        if self.detection_limit and len(loss_history) > 0:
            if (loss_history[-1] / min(loss_history) > self.detection_coef) or math.isnan(loss_history[-1]):
                return True
        return False

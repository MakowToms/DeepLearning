"""Potwierdzam samodzielność powyższej pracy oraz niekorzystanie przeze mnie z niedozwolonych źródeł. ~Mateusz Bąkała"""

import numpy as np


class Stopper:
    """Takes care of monitoring whether to run another iteration. Rather meager now, but no need to complicate it
    at the moment. Can possibly take multiple stop conditions."""
    def __init__(self):
        # possible stop conditions
        self.max_iter = np.infty
        # internal variables
        self.iteration = 1

    def stop(self, algorithm=None):
        """Responds if algorithm should stop now or is allowed to run another iteration. Takes algorithm as a parameter
        to allow extending it to cover cases where it is to stop after reaching certain solution value or something."""
        if self.iteration >= self.max_iter:
            return True
        self.iteration += 1
        return False

    def set_max_iter(self, max_iter):
        """Set maximum iteration limit. Isn't exclusive with other constraints."""
        self.max_iter = max_iter
        return self

import numpy as np


class WeightInitializer:
    def __init__(self, weight_init, bias_init):
        self.weight_init = weight_init
        self.bias_init = bias_init

    def initialize(self, layer):
        """Generates initial weights and biases given layer and input layer data."""
        layer.weights = self.weight_init(layer)
        layer.biases = self.bias_init(layer)


zero_init = WeightInitializer(
    lambda layer: np.array([[0.0 for _ in range(layer.input.size)] for _ in range(layer.size)]),
    lambda layer: np.array([[0.0] for _ in range(layer.size)])
)
uniform_init = WeightInitializer(
    lambda layer: np.array([[np.random.uniform(0, 1) for _ in range(layer.input.size)] for _ in range(layer.size)]),
    lambda layer: np.array([[np.random.uniform(0, 1)] for _ in range(layer.size)])
)
normal_init = WeightInitializer(
    lambda layer: np.array([[np.random.normal(0, 1) for _ in range(layer.input.size)] for _ in range(layer.size)]),
    lambda layer: np.array([[np.random.normal(0, 1)] for _ in range(layer.size)])
)

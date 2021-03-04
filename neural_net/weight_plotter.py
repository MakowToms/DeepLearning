import numpy as np

import VisualizeNN.VisualizeNN as VisNN


def plot_weights(nn, plot_name):
    sizes = [layer.size for layer in nn.layers]
    weights = [np.array(layer.weights).T for layer in nn.layers[1:]]
    network = VisNN.DrawNN(sizes, weights, plot_name)
    network.draw()

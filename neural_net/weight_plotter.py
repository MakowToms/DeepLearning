import numpy as np

import VisualizeNN.VisualizeNN as VisNN


def plot_weights(nn, file_name):
    sizes = [layer.size for layer in nn.layers]
    weights = [np.array(layer.weights).T for layer in nn.layers[1:]]
    network = VisNN.DrawNN(sizes, "Edge weights", weights, file_name)
    network.draw()


def plot_errors(nn, file_name):
    sizes = [layer.size for layer in nn.layers]
    errors = [np.array(layer.last_error).T * 10 for layer in nn.layers[1:]]
    network = VisNN.DrawNN(sizes, "Backpropagated errors", errors, file_name)
    network.draw(sparse_labels=False)

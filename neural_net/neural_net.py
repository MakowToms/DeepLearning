import math

import matplotlib.pyplot as plt
import numpy as np

from neural_net.budget import Budget
from neural_net.losses import MSE
from neural_net.optimizers import no_optimizer
from neural_net.regularizations import no_regularization
from neural_net.weights import zero_init


def split(data, batch_size):
    """Splits data into batches."""
    # implemented because of that transposition; should I get rid of it?
    return [np.transpose(batch) for batch in np.split(np.transpose(data), range(batch_size, data.shape[1], batch_size))]


class InputLayer:
    def __init__(self, size):
        self.size = size
        self.values = None

    def set_data(self, data):
        self.values = data


class Layer:
    def __init__(self, size, activation):
        # layer parameters
        self.size = size
        self.input = None
        self.weights = None
        self.biases = None
        # activations
        # setting it like that allows manually setting different activations for different neurons within layer
        self.activations = np.array([activation] * size)
        # computed values
        self.weighted_input = None
        self.values = None
        # backpropagation
        self.local_gradient = None
        self.weights_diff = None
        self.biases_diff = None
        # optimizer
        self.optimizer = no_optimizer
        self.weights_momentum = None
        self.biases_momentum = None

    def __add_input__(self, inp, weight_init):
        """Passes reference to input Layer object and initializes weights."""
        self.input = inp
        weight_init.initialize(self)
        self.weights_momentum = np.array([[0 for _ in range(inp.size)] for _ in range(self.size)])
        self.biases_momentum = np.array([[0] for _ in range(self.size)])

    def __predict__(self):
        # compute weighted input (input activations + biases), without applying activation function
        self.weighted_input = np.array(self.weights.dot(self.input.values)) + np.array(self.biases)
        # initialize returned values with zeros
        self.values = np.zeros(self.weighted_input.shape)
        # apply activation function to neuron
        # iterate over observations
        for i in range(self.weighted_input.shape[1]):
            # iterate over neurons within layer
            for j in range(self.weighted_input.shape[0]):
                # apply neuron activation function to activation values of this neuron and whole layer
                self.values[j, i] = self.activations[j].get_function()(self.weighted_input[j, i], self.weighted_input[:, i])
        # NOTE: there used to be vectorized function, but passing whole layer made it too complicated

    def __backpropagate__(self, weighted_error=None):
        """Makes some mathematical magic and possibly generates uncatched computational errors."""
        values = np.zeros(self.weighted_input.shape)
        # apply derivative of activation function to neuron
        # iterate over observations
        for i in range(self.weighted_input.shape[1]):
            # iterate over neurons within layer
            for j in range(self.weighted_input.shape[0]):
                # apply derivative of neuron activation function to activation values of this neuron and whole layer
                values[j, i] = self.activations[j].get_derivative()(self.weighted_input[j, i], self.weighted_input[:, i])
        self.local_gradient = np.multiply(values, weighted_error)
        # optimization (computing weight and bias differences mostly)
        self.optimizer.optimize(self)


class NeuralNet:
    def __init__(self, size, weight_init=zero_init, seed=None):
        if seed:
            np.random.seed(seed)
        self.layers = [InputLayer(size)]
        self.loss = MSE
        self.regularization = no_regularization
        self.loss_history = None
        self.weight_init = weight_init
        self.budget = Budget()

    def __backpropagate__(self, y, learning_rate=0.001):
        # compute weight and bias changes for every layer starting with the last
        self.layers[-1].__backpropagate__(self.get_result() - y)
        for layer, next_layer in zip(reversed(self.layers[1:-1]), reversed(self.layers[2:])):
            weighted_error = np.transpose(next_layer.weights).dot(next_layer.local_gradient)
            layer.__backpropagate__(weighted_error)
        # apply computed changes to weights and biases
        # again, starting with the last layer (I forgot why, seems like it doesn't matter)
        for layer in self.layers[1:]:
            # modified weight update because of regularization support
            layer.weights -= learning_rate * (layer.weights_diff + self.regularization.update_weights(layer))
            layer.biases -= learning_rate * layer.biases_diff

    def add_layer(self, layer):
        """Appends layer to NeuralNet object. Should be given Layer object as parameter."""
        layer.__add_input__(self.layers[-1], self.weight_init)
        self.layers.append(layer)
        return self

    def set_optimizer(self, optimizer):
        """Adds optimizer to backpropagation algorithm. Should be an object of Optimizer class."""
        # each layer could possibly has its own optimizer
        # but usually we want to set references to one optimizer for every layer
        # so we can tune optimizer's parameters from outside the net
        for layer in self.layers[1:]:
            layer.optimizer = optimizer
        return self

    def set_regularization(self, regularization):
        self.regularization = regularization
        return self

    def set_loss(self, loss):
        """@Deprecated, might cause backpropagation to work incorrectly."""
        self.loss = loss
        return self

    def get_result(self):
        """Returns last predicted values."""
        return self.layers[-1].values

    def get_loss(self, y):
        weight_loss = 0
        for layer in self.layers[1:]:
            weight_loss += self.regularization.compute_loss(layer)
        return self.loss.compute_loss(self.get_result(), y) + weight_loss

    def train(self, data, y, learning_rate=0.001, batch_size=10, verbose=True):
        """Calls backpropagation algorithm with MSE loss function to fit weights."""

        # data and y have now each observation as column
        data = np.transpose(data)
        y = np.transpose(y)

        # initialize some variables (loss_history with inf because of finding minimum later)
        self.loss_history = []
        # initial prediction
        self.predict(data)
        loss = self.get_loss(y)
        # check if any dimension returns NaN
        if any(map(math.isnan, loss)):
            loss = math.nan
        else:
            loss = loss.sum()
        while not self.budget.finished(self.loss_history):
            print("Epoch: {}".format(self.budget.epoch + 1))
            # split data into batches
            data, y = self.__shuffle__(data, y)
            batches = zip(split(data, batch_size), split(y, batch_size))
            for index, batch in enumerate(batches):
                # make prediction for given batch
                self.predict(batch[0])
                # run backpropagation with given batch
                self.__backpropagate__(batch[1], learning_rate)
                # compute loss for whole dataset
                self.predict(data)
                loss = self.get_loss(y)
                # check if any dimension returns NaN
                if any(map(math.isnan, loss)):
                    loss = math.nan
                else:
                    loss = loss.sum()
                if verbose:
                    print("Batch {0}/{1}".format(index + 1, math.ceil(y.shape[1] / batch_size)))
                    print("Loss: {}".format(loss))
            print("==========================")
            # save final loss for given epoch
            self.loss_history.append(loss)
            self.budget.epoch += 1

    def predict(self, data):
        """Uses internal weights to predict answer for given data."""
        self.layers[0].set_data(data)
        for layer in self.layers[1:]:
            layer.__predict__()

    def plot_prediction(self, x, y):
        """Works for 1D data, that is - one x column and one y column."""
        self.predict(x)
        plt.scatter(x, np.transpose(self.get_result()), label="prediction")
        plt.scatter(x, y, label="true")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.title("Prediction comparison")

    @staticmethod
    def __shuffle__(data, y):
        s = np.arange(data.shape[1])
        np.random.shuffle(s)
        return data[:, s], y[:, s]

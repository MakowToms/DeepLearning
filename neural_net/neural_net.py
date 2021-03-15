import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neural_net.budget import Budget
from neural_net.losses import MSE, accuracy
from neural_net.optimizers import no_optimizer
from neural_net.regularizations import no_regularization
from neural_net.weight_plotter import plot_weights, plot_errors
from neural_net.weights import zero_init

# Taken from https://stackoverflow.com/questions/15713279/calling-pylab-savefig-without-display-in-ipython#15713545
matplotlib.use('Agg')


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
        self.activation = activation
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
        # error plot
        self.last_error = None

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
        if self.activation.name == 'softmax':
            for i in range(self.weighted_input.shape[1]):
                # iterate over neurons within layer
                for j in range(self.weighted_input.shape[0]):
                    # apply neuron activation function to activation values of this neuron and whole layer
                    self.values[j, i] = self.activation.get_function()(self.weighted_input[j, i], self.weighted_input[:, i])
        else:
            self.values = self.activation.get_function()(self.weighted_input, "whatever")
        # NOTE: there used to be vectorized function, but passing whole layer made it too complicated

    def __backpropagate__(self, weighted_error=None):
        """Makes some mathematical magic and possibly generates uncatched computational errors."""
        values = np.zeros(self.weighted_input.shape)
        # apply derivative of activation function to neuron
        # iterate over observations
        if self.activation.name == 'softmax':
            for i in range(self.weighted_input.shape[1]):
                # iterate over neurons within layer
                for j in range(self.weighted_input.shape[0]):
                    # apply derivative of neuron activation function to activation values of this neuron and whole layer
                    values[j, i] = self.activation.get_derivative()(self.weighted_input[j, i], self.weighted_input[:, i])
        else:
            values = self.activation.get_derivative()(self.weighted_input, "whatever")

        self.local_gradient = np.multiply(values, weighted_error)

        # saves weighted error for each edge to visualize it
        self.last_error = self.local_gradient.dot(np.transpose(self.input.values))
        # optimization (computing weight and bias differences mostly)
        self.optimizer.optimize(self)


class NeuralNet:
    def __init__(self, size, *, weight_init=zero_init, seed=None, name="neural_network", visualize=False, is_regression=True):
        if seed:
            np.random.seed(seed)
        self.layers = [InputLayer(size)]
        self.loss = MSE
        self.regularization = no_regularization
        self.weight_init = weight_init
        self.name = name
        self.visualize = visualize
        self.budget = Budget()
        self.is_regression = is_regression

    def __backpropagate__(self, y, learning_rate=0.001):
        # compute weight and bias changes for every layer starting with the last
        self.layers[-1].__backpropagate__(self.loss.compute_derivative(self.get_result(), y))
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

    def get_loss_train(self):
        return self.loss_history_train

    def get_loss_test(self):
        return self.loss_history_test

    def get_MSE_train(self):
        return self.MSE_train

    def get_MSE_test(self):
        return self.MSE_test

    def save_metrics(self, data, y, x_test, y_test):
        self.predict(x_test)
        loss = self.get_loss(y_test)
        self.loss_history_test.append(loss)
        print("Loss: {}".format(self.get_loss(y_test)))
        print("==========================")

        self.predict(data.T)
        loss = self.get_loss(y)
        self.loss_history_train.append(loss)

        if self.is_regression:
            self.MSE_test.append(MSE.compute_loss(self.predict(x_test), np.transpose(y_test)))
            self.MSE_train.append(MSE.compute_loss(self.predict(np.transpose(data)), np.transpose(y)))
        else:
            self.MSE_test.append(accuracy(self.predict(x_test), np.transpose(y_test)))
            self.MSE_train.append(accuracy(self.predict(np.transpose(data)), np.transpose(y)))

    def fit(self, data, y, x_test, y_test, learning_rate=0.001, batch_size=10):
        for layer, previous_layer in zip(self.layers[1:], self.layers[:-1]):
            layer.__add_input__(previous_layer, self.weight_init)
        self.budget.epoch = 0
        self._fit(data, y, x_test, y_test, learning_rate, batch_size)

    def partial_fit(self, data, y, x_test, y_test, learning_rate=0.001, batch_size=10):
        self._fit(data, y, x_test, y_test, learning_rate, batch_size)

    def _fit(self, data, y, x_test, y_test, learning_rate, batch_size):
        """Calls backpropagation algorithm with given loss function to fit weights."""

        # data and y have now each observation as column
        data = np.transpose(data)
        y = np.transpose(y)
        y_test = np.transpose(y_test)

        # initialize some variables (loss_history with inf because of finding minimum later)
        self.loss_history_train = []
        self.loss_history_test = []
        self.MSE_train = []
        self.MSE_test = []

        while not self.budget.finished(self.loss_history_test):
            print("Epoch: {}".format(self.budget.epoch + 1))
            # split data into batches
            data, y = self.__shuffle__(data, y)
            batches = zip(split(data, batch_size), split(y, batch_size))
            for index, batch in enumerate(batches):
                # make prediction for given batch
                self.predict(np.transpose(batch[0]))
                # run backpropagation with given batch
                self.__backpropagate__(batch[1], learning_rate)
                print("Batch {0}/{1}".format(index + 1, math.ceil(y.shape[1] / batch_size)), end="\r")

            self.save_metrics(data, y, x_test, y_test)
            if not os.path.exists("plots"):
                os.mkdir("plots")
            if self.visualize:
                plot_weights(self, "plots/{0}_weights_{1}.png".format(self.name, self.budget.epoch + 1))
                plot_errors(self, "plots/{0}_errors_{1}.png".format(self.name, self.budget.epoch + 1))
            self.budget.epoch += 1

    def predict(self, data):
        """Uses internal weights to predict answer for given data."""
        self.layers[0].set_data(np.transpose(data))
        for layer in self.layers[1:]:
            layer.__predict__()
        return np.transpose(self.get_result())

    def predict_with_threshold(self, data, threshold=0.5):
        self.layers[0].set_data(np.transpose(data))
        for layer in self.layers[1:]:
            layer.__predict__()
        results = np.transpose(self.get_result())
        if results.shape[1] == 1:
            return results
        results_1d = np.zeros(results.shape[0])
        for i in range(results.shape[1]):
            results_1d[results[:, i] > threshold] = i
        return results_1d

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

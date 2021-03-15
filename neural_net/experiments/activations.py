import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from neural_net.activations import softmax, tanh, sigmoid, ReLU, linear
from neural_net.losses import LogLoss, MSE
from neural_net.neural_net import NeuralNet, Layer
from neural_net.optimizers import RMSProp
from neural_net.regularizations import L1_regularization
from neural_net.weights import uniform_init
from neural_net.plot import Plotter
from neural_net.datasets import datasets

np.random.seed(123)
for dataset in datasets:
    nns = []
    output_activation = softmax if dataset.task_type == "classification" else linear
    n_output_neurons = dataset.y_train.shape[1] if dataset.task_type == "classification" else 1
    loss = LogLoss if dataset.task_type == "classification" else MSE
    error_name = 'Accuracy' if dataset.task_type == "classification" else 'MSE'
    activation_error = []
    for activation in [linear, ReLU, sigmoid, tanh]:
        nn = NeuralNet(dataset.x_train.shape[1], weight_init=uniform_init, name=activation.name,
                       is_regression=dataset.task_type == "regression") \
            .add_layer(Layer(20, activation)) \
            .add_layer(Layer(n_output_neurons, output_activation)) \
            .set_optimizer(RMSProp.set_params({"coef": 0.9})) \
            .set_regularization(L1_regularization.set_params({"coef": 0.0001})) \
            .set_loss(loss)
        nn.budget.set_epoch_limit(100).set_detection_limit(2)
        n = 10
        errors = [np.empty(n), np.empty(n), np.empty(n), np.empty(n)]
        for i in range(n):
            nn.fit(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, learning_rate=0.01, batch_size=32)
            errors[0][i] = nn.get_loss_train()[-1]
            errors[1][i] = nn.get_loss_test()[-1]
            errors[2][i] = nn.get_MSE_train()[-1]
            errors[3][i] = nn.get_MSE_test()[-1]
        nns.append(nn)
        activation_error.append(errors)

    # plot error boxplot
    plotter = Plotter(dataset.x_test, dataset.y_test, nns)
    plt.subplots(2, 2)
    plt.subplot(2, 2, 1)
    plotter.boxplot_of_errors(activation_error, 0, 'Loss train')
    plt.subplot(2, 2, 2)
    plotter.boxplot_of_errors(activation_error, 1, 'Loss test')
    plt.subplot(2, 2, 3)
    plotter.boxplot_of_errors(activation_error, 2, f'{error_name} train')
    plt.subplot(2, 2, 4)
    plotter.boxplot_of_errors(activation_error, 3, f'{error_name} test')

    # plot errors from last evaluation
    plt.subplots(2, 2)
    plt.subplot(2, 2, 1)
    plotter.plot_measure_results_data(NeuralNet.get_loss_test, 'Loss test', show=False)
    plt.subplot(2, 2, 2)
    plotter.plot_measure_results_data(NeuralNet.get_loss_train, 'Loss train', show=False)
    plt.subplot(2, 2, 3)
    plotter.plot_measure_results_data(NeuralNet.get_MSE_test, f'{error_name} test', show=False)
    plt.subplot(2, 2, 4)
    plotter.plot_measure_results_data(NeuralNet.get_MSE_train, f'{error_name} train', show=False)
    plt.show()

    # plot data 1d or 2d
    if dataset.task_type == "classification":
        plt.subplots(2, 2)
        for i in range(4):
            plt.subplot(2, 2, i+1)
            plotter.plot_data_2d(i)
    else:
        plt.subplots(1, 1)
        plotter.plot_data_1d()


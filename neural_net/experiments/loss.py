import os

import matplotlib.pyplot as plt
import numpy as np

from neural_net.activations import softmax, tanh, sigmoid, ReLU, linear
from neural_net.datasets import Dataset
from neural_net.losses import MSE, hinge, LogLoss, MAE
from neural_net.neural_net import NeuralNet, Layer
from neural_net.optimizers import RMSProp
from neural_net.plot import Plotter
from neural_net.regularizations import L1_regularization
from neural_net.weights import uniform_init
from utils.latex import save_errors_to_latex

if not os.path.exists("plots"):
    os.mkdir("plots")
if not os.path.exists("plots/loss"):
    os.mkdir("plots/loss")
if not os.path.exists("tables"):
    os.mkdir("tables")
if not os.path.exists("tables/loss"):
    os.mkdir("tables/loss")

sizes = [1000]
names = [("simple", "classification"), ("three_gauss", "classification"),
         ("activation", "regression"), ("cube", "regression")]
datasets = [Dataset(name, task_type, size) for name, task_type in names for size in sizes]

for dataset in datasets:
    np.random.seed(123)
    nns = []
    output_activation = softmax if dataset.task_type == "classification" else linear
    n_output_neurons = dataset.y_train.shape[1] if dataset.task_type == "classification" else 1
    error_name = 'Accuracy' if dataset.task_type == "classification" else 'MSE'
    first_layer_size = 10 if dataset.task_type == "classification" else 50
    n_epochs = 20 if dataset.name == "three_gauss" else 100
    losses = [hinge, hinge, LogLoss, LogLoss] if dataset.task_type == "classification" else [MSE, MSE, MAE, MAE]
    activations = [tanh, sigmoid, tanh, sigmoid]
    loss_acctivation_error = []
    for loss, activation in zip(losses, activations):
        nn = NeuralNet(dataset.x_train.shape[1], weight_init=uniform_init, name=f"{loss.name}, {activation.name}",
                       is_regression=dataset.task_type == "regression") \
            .add_layer(Layer(first_layer_size, activation)) \
            .add_layer(Layer(n_output_neurons, output_activation)) \
            .set_optimizer(RMSProp.set_params({"coef": 0.9})) \
            .set_regularization(L1_regularization.set_params({"coef": 0.0001})) \
            .set_loss(loss)
        nn.budget.set_epoch_limit(n_epochs).set_detection_limit(3)
        n = 10
        errors = [np.empty(n), np.empty(n), np.empty(n), np.empty(n)]
        for i in range(n):
            nn.fit(dataset.x_train, dataset.y_train, dataset.x_test, dataset.y_test, learning_rate=0.01, batch_size=32)
            errors[0][i] = nn.get_loss_train()[-1]
            errors[1][i] = nn.get_loss_test()[-1]
            errors[2][i] = nn.get_MSE_train()[-1]
            errors[3][i] = nn.get_MSE_test()[-1]
        nns.append(nn)
        loss_acctivation_error.append(errors)
    save_errors_to_latex(loss_acctivation_error, [f"loss={loss.name}, activation={activation.name}" for loss, activation in zip(losses, activations)],
                         ["{0} on train".format(loss.name), "{0} on test".format(loss.name),
                          "{0} on train".format(error_name), "{0} on test".format(error_name)],
                         f"{str.capitalize(dataset.name)} dataset of size {dataset.size}",
                         f"tab:loss:{dataset.name}_{dataset.size}",
                         "tables/loss/{0}_{1}.txt".format(dataset.name, dataset.size),
                         shrink=True)

    # plot error boxplot
    plotter = Plotter(dataset.x_test, dataset.y_test, nns)
    plt.subplots(2, 2, figsize=(6.4, 4.8))
    plt.subplots_adjust(wspace=0.3, hspace=1.1, bottom=0.3)
    boxplot_labels = [f"{loss.name}, {activation.name}" for loss, activation in zip(losses, activations)]
    plt.subplot(2, 2, 1)
    plotter.boxplot_of_errors(loss_acctivation_error, 2, f'{error_name} train', show=False, labels=boxplot_labels, rotation=30)
    plt.subplot(2, 2, 2)
    plotter.boxplot_of_errors(loss_acctivation_error, 3, f'{error_name} test', show=False, labels=boxplot_labels, rotation=30)
    plt.subplot(2, 2, 3)
    plotter.boxplot_of_errors(loss_acctivation_error, 0, 'Loss train', show=False, labels=boxplot_labels, rotation=30)
    plt.subplot(2, 2, 4)
    plotter.boxplot_of_errors(loss_acctivation_error, 1, 'Loss test', show=False, labels=boxplot_labels, rotation=30)
    plt.savefig("plots/loss/{0}_{1}_loss_boxplot.png".format(dataset.name, dataset.size),
                dpi=100, bbox_inches="tight")

    # plot errors from last evaluation
    plt.subplots(2, 2, figsize=(6.4, 4.8))
    plt.subplots_adjust(wspace=0.3, hspace=0.6)
    plt.subplot(2, 2, 1)
    plotter.plot_measure_results_data(NeuralNet.get_MSE_train, f'{error_name} train', show=False)
    plt.subplot(2, 2, 2)
    plotter.plot_measure_results_data(NeuralNet.get_MSE_test, f'{error_name} test', show=False)
    plt.subplot(2, 2, 3)
    plotter.plot_measure_results_data(NeuralNet.get_loss_train, 'Loss train', show=False)
    plt.subplot(2, 2, 4)
    plotter.plot_measure_results_data(NeuralNet.get_loss_test, 'Loss test', show=False)
    plt.savefig("plots/loss/{0}_{1}_loss_history.png".format(dataset.name, dataset.size),
                dpi=100, bbox_inches="tight")

    # plot data 1d or 2d
    if dataset.task_type == "classification":
        plt.subplots(2, 2, figsize=(6.4, 4.8))
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        for i, loss, activation in zip([0, 1, 2, 3], losses, activations):
            plt.subplot(2, 2, i+1)
            plotter.plot_data_2d(i, title=f"loss={loss.name}, activation={activation.name}", show=False)
            plt.savefig("plots/loss/{0}_{1}_points.png".format(dataset.name, dataset.size),
                        dpi=100, bbox_inches="tight")
    else:
        plt.subplots(1, 1)
        plotter.plot_data_1d(show=False)
        plt.savefig("plots/loss/{0}_{1}_values.png".format(dataset.name, dataset.size),
                    dpi=100, bbox_inches="tight")


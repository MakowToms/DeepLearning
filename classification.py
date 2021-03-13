import pandas as pd

from neural_net.activations import softmax, tanh
from neural_net.losses import LogLoss, MSE
from neural_net.neural_net import NeuralNet, Layer
from neural_net.optimizers import RMSProp
from neural_net.regularizations import L1_regularization
from neural_net.weights import uniform_init
from neural_net.prepare_dataset import x_y_split

from neural_net.plot import Plotter

train = pd.read_csv('datasets/neural_net/classification/data.simple.train.1000.csv')
test = pd.read_csv('datasets/neural_net/classification/data.simple.test.1000.csv')
x_train, y_train = x_y_split(train, 'cls')
x_test, y_test = x_y_split(test, 'cls')

nn = NeuralNet(2, weight_init=uniform_init)\
    .add_layer(Layer(20, tanh))\
    .add_layer(Layer(2, softmax))\
    .set_optimizer(RMSProp.set_params({"coef": 0.1}))\
    .set_regularization(L1_regularization.set_params({"coef": 0.05}))\
    .set_loss(LogLoss)
nn.budget.set_epoch_limit(3).set_detection_limit(1.3)
nn.fit(x_train, y_train, x_test, y_test, learning_rate=0.02, batch_size=32)

print(f'MSE: {MSE.compute_loss(nn.predict(x_train), y_train)}')

plotter = Plotter(x_test, y_test, [nn])
plotter.plot_data_1d()
plotter.plot_measure_results_data(NeuralNet.get_loss_history, "LogLoss")
plotter.plot_measure_results_data(NeuralNet.get_MSE_test, "MSE test")


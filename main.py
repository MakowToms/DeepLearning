import numpy as np
import pandas

from neural_net.activations import sigmoid, linear
from neural_net.neural_net import NeuralNet, Layer
from neural_net.optimizers import momentum
from neural_net.regularizations import L1_regularization
from neural_net.weights import uniform_init

train = pandas.read_csv('datasets/neural_net/regression/data.cube.test.100.csv')
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1:]

nn = NeuralNet(1, uniform_init)\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(15, sigmoid))\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(1, linear))\
    .set_optimizer(momentum.set_params({"coef": 0.05}))\
    .set_regularization(L1_regularization.set_params({"coef": 0.05}))
nn.budget.set_epoch_limit(50).set_detection_limit(1.3)
nn.train(x_train, y_train, learning_rate=0.02, batch_size=16)

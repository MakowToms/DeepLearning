import pandas as pd

from neural_net.activations import sigmoid, linear, softmax
from neural_net.losses import MAE, hinge
from neural_net.neural_net import NeuralNet, Layer
from neural_net.optimizers import momentum
from neural_net.regularizations import L1_regularization
from neural_net.weights import uniform_init
from neural_net.prepare_dataset import x_y_split


train = pd.read_csv('datasets/neural_net/regression/data.cube.test.100.csv')
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1:]

nn = NeuralNet(1, weight_init=uniform_init)\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(15, sigmoid))\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(1, linear))\
    .set_optimizer(momentum.set_params({"coef": 0.05}))\
    .set_regularization(L1_regularization.set_params({"coef": 0.05}))\
    .set_loss(MAE)
nn.budget.set_epoch_limit(10).set_detection_limit(1.3)
nn.train(x_train, y_train, learning_rate=0.02, batch_size=32)

# Classification
train = pd.read_csv('datasets/neural_net/classification/data.simple.train.100.csv')
x_train, y_train = x_y_split(train, 'cls')

nn = NeuralNet(2, weight_init=uniform_init)\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(15, sigmoid))\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(2, sigmoid))\
    .set_optimizer(momentum.set_params({"coef": 0.05}))\
    .set_loss(hinge)
nn.budget.set_epoch_limit(50).set_detection_limit(1.3)
nn.train(x_train, y_train, learning_rate=0.02, batch_size=8)

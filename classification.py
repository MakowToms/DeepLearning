import pandas as pd
import numpy as np

from neural_net.activations import softmax, tanh
from neural_net.losses import LogLoss
from neural_net.neural_net import NeuralNet, Layer
from neural_net.optimizers import RMSProp
from neural_net.regularizations import L1_regularization
from neural_net.weights import uniform_init
from neural_net.prepare_dataset import x_y_split

train = pd.read_csv('datasets/neural_net/classification/data.simple.train.1000.csv')
x_train, y_train = x_y_split(train, 'cls')

nn = NeuralNet(2, uniform_init)\
    .add_layer(Layer(20, tanh))\
    .add_layer(Layer(2, softmax))\
    .set_optimizer(RMSProp.set_params({"coef": 0.1}))\
    .set_regularization(L1_regularization.set_params({"coef": 0.05}))\
    .set_loss(LogLoss)
nn.budget.set_epoch_limit(10).set_detection_limit(1.3)
nn.train(x_train, y_train, learning_rate=0.02, batch_size=32)

mse = np.mean(np.abs(nn.predict(np.transpose(x_train)) - y_train.transpose()))

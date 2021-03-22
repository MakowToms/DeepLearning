import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from neural_net.activations import tanh, softmax, linear, sigmoid
from neural_net.losses import hinge, MSE
from neural_net.neural_net import NeuralNet, Layer
from neural_net.optimizers import RMSProp
from neural_net.prepare_dataset import x_y_split
from neural_net.regularizations import L1_regularization
from neural_net.weights import uniform_init

train = pd.read_csv('datasets/neural_net/mnist/train.csv')
x_test = np.array(pd.read_csv('datasets/neural_net/mnist/test.csv'))
x_train, y_train = x_y_split(train, 'label')
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=35000, random_state=123)

np.random.seed(123)


def accuracy(result, y_numeric):
    return np.sum(y_numeric.transpose()==result)/y_numeric.shape[0]


nn = NeuralNet(x_train.shape[1], weight_init=uniform_init, name="mnist", is_regression=False) \
    .add_layer(Layer(100, sigmoid)) \
    .add_layer(Layer(y_train.shape[1], softmax)) \
    .set_loss(MSE)
    # .set_optimizer(RMSProp.set_params({"coef": 0.9})) \
    # .set_regularization(L1_regularization.set_params({"coef": 0.0001})) \
nn.budget.set_epoch_limit(10)
nn.fit(x_train/255, y_train, x_val/255, y_val, learning_rate=0.002, batch_size=128)

res = nn.predict(x_val/255)
res = np.argmax(res, axis=1)
print(accuracy(res, np.argmax(y_val, axis=1)))

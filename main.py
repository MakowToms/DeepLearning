import pandas

from neural_net.activations import sigmoid, linear, softmax
from neural_net.losses import MAE, hinge
from neural_net.neural_net import NeuralNet, Layer
from neural_net.optimizers import momentum
from neural_net.regularizations import L1_regularization
from neural_net.weights import uniform_init


def one_hot(df, column):
    dummies = pandas.get_dummies(df[column], prefix=column)
    df_without_column = df.drop([column], axis=1)
    return df_without_column, dummies


train = pandas.read_csv('datasets/neural_net/regression/data.cube.test.100.csv')
x_train = train.iloc[:, :-1]
y_train = train.iloc[:, -1:]

nn = NeuralNet(1, uniform_init)\
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
train = pandas.read_csv('datasets/neural_net/classification/data.simple.train.100.csv')
x_train, y_train = one_hot(train, 'cls')

nn = NeuralNet(2, uniform_init)\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(15, sigmoid))\
    .add_layer(Layer(10, sigmoid))\
    .add_layer(Layer(2, linear))\
    .set_optimizer(momentum.set_params({"coef": 0.05}))\
    .set_loss(hinge)
nn.budget.set_epoch_limit(10).set_detection_limit(1.3)
nn.train(x_train, y_train, learning_rate=0.02, batch_size=8)

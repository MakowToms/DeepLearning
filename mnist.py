import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from neural_net.prepare_dataset import x_y_split

train = pd.read_csv('datasets/neural_net/mnist/train.csv')
x_test = np.array(pd.read_csv('datasets/neural_net/mnist/test.csv'))
x_train, y_train = x_y_split(train, 'label')
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=35000, random_state=123)

np.random.seed(123)

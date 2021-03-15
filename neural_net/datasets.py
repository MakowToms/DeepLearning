import pandas as pd

from neural_net.prepare_dataset import x_y_split, x_y_split_by_index


class Dataset:
    def __init__(self, name, task_type, size):
        train = pd.read_csv('datasets/neural_net/{0}/data.{1}.train.{2}.csv'.format(task_type, name, size))
        test = pd.read_csv('datasets/neural_net/{0}/data.{1}.test.{2}.csv'.format(task_type, name, size))

        self.name = name
        if task_type == "classification":
            self.x_train, self.y_train = x_y_split(train, 'cls')
            self.x_test, self.y_test = x_y_split(test, 'cls')
        else:
            self.x_train, self.y_train = x_y_split_by_index(train, -1)
            self.x_test, self.y_test = x_y_split_by_index(test, -1)
        self.task_type = task_type


sizes = [100, 500, 1000, 10000]
names = [("simple", "classification"), ("three_gauss", "classification"),
         ("activation", "regression"), ("cube", "regression")]
datasets = [Dataset(name, task_type, size) for name, task_type in names for size in sizes]

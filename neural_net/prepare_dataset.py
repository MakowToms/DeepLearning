import pandas as pd
import numpy as np


def x_y_split(df, column):
    dummies = pd.get_dummies(df[column], prefix=column)
    df_without_column = df.drop([column], axis=1)
    return np.array(df_without_column), np.array(dummies)


def x_y_split_by_index(df, index):
    return np.array(df.iloc[:, :index]), np.array(df.iloc[:, index:])

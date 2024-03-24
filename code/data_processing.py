import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def as_discrete(col):
    n = len(col)
    new_col = [0] * n
    for i in range(n):
        if col[i] == b"0":
            new_col[i] = 0
        else:
            new_col[i] = 1
    return pd.DataFrame(new_col)


def get_Xy(df):
    X = df.iloc[:, 0 : len(df) - 1]
    y = as_discrete(df.iloc[:, -1])
    return X, y


def med_impute(df, y):
    # remove columns with more than 40% values being null
    thd1 = df.shape[0] * 0.4
    cols = df.columns[df.isnull().sum() < thd1]
    df = df[cols]

    # remove rows with more than 50% values being null
    thd2 = df.shape[1] * 0.5
    y = y[df.isnull().sum(axis=1) <= thd2]
    df = df[df.isnull().sum(axis=1) <= thd2]

    # median imputation for null values
    df.fillna(df.median())

    return df, y


def normalise(df):
    scaler = MinMaxScaler()
    X_scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

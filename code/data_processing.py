import numpy as np
import pandas as pd
from sklearn import minmaxscaler

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
    X = df_origin.iloc[:, 0:len(df)-1]
    y = as_discrete(df.iloc[:, -1])
    return X, y

def null_removal(df):
    # Define a function to convert binary columns to discrete
    def as_discrete(col):
        n = len(col)
        new_col = [0] * n
        for i in range(n):
            if col[i] == b'0':
                new_col[i] = 0
            else:
                new_col[i] = 1
        return pd.DataFrame(new_col)

    # Extract features (X) and target (y)
    X = df.iloc[:, :-1]
    y = as_discrete(df.iloc[:, -1])

    # Remove rows where more than half of the entries are null
    threshold = X.shape[1] / 2  # Total number of columns divided by 2
    X_clean = X[X.isnull().sum(axis=1) <= threshold]

    # Impute missing values with the median
    X_imputed = X_clean.fillna(X_clean.median())
    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)

    # without changing column names
    x_scaled_origin = scaler.fit_transform(X_imputed)
    X_scaled_df_origin = pd.DataFrame(x_scaled_origin, columns=X_imputed.columns)
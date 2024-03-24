import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

'''
functions starting with df_ can generate a processed dataframe directly
'''

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
    X = df.iloc[:, 0:len(df)-1]
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


def df_null_removal(df):
    # Extract features (X) and target (y)
    X, y = get_Xy(df)

    # Remove null values and impute missing values
    X_imputed, y = med_impute(X, y)

    # Scale the imputed data
    X_scaled_df = normalise(X_imputed)

    return pd.concat([X_scaled_df, y], axis=1)


def drop_high_corr(df, threshold=0.7):

    correlation_matrix = df.corr()
    high_cor = []
    dropped_features = []

    # Iterate through the correlation matrix to find highly correlated pairs
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > threshold:
                if correlation_matrix.columns[j] != correlation_matrix.columns[i]:
                    high_cor.append([
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ])

    # Iterate through the list of highly correlated pairs
    for pair in high_cor:
        feature1, feature2, correlation = pair

        # Check if either of the features in the pair has already been dropped
        if feature1 not in dropped_features and feature2 not in dropped_features:
            # Check if the feature exists in the DataFrame before attempting to drop it
            if feature2 in df.columns:
                # Drop one of the correlated features from the dataset
                # Here, we arbitrarily choose to drop the second feature in the pair
                df.drop(feature2, axis=1, inplace=True)
                dropped_features.append(feature2)
            else:
                print(f"Feature '{feature2}' not found in the DataFrame.")

    return df

def df_null_corr_process(df):
    return drop_high_corr(df_null_removal(df))
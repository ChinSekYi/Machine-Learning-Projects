import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import arff
from sklearn.feature_selection import SelectKBest, f_classif

import data_processing

data = arff.loadarff("../data/3year.arff")
df = pd.DataFrame(data[0])
df_origin = df.copy()


def get_df_with_top_k_features(k_features, df_x, df_y):
    print(df.shape)  # (4885, 64)
    print(len(df.columns))

    # df_x, df_y = get_Xy(df)
    # df_x = df.iloc[:, :-1].to_numpy()
    # df_y = df.iloc[:, -1].to_numpy()

    # define feature selection
    fs = SelectKBest(score_func=f_classif, k=k_features)

    # apply feature selection
    X_selected = fs.fit_transform(df_x, df_y)

    # Take the features with the highest F-scores
    fs_scores_array = np.array(fs.scores_)

    # Get the indices that would sort the array in descending order
    sorted_indices_desc = np.argsort(fs_scores_array)[::-1]

    # Take the top k indices
    top_indices = sorted_indices_desc[:k_features]
    selected_indices = np.append(top_indices, 63)

    selected_columns = df.iloc[:, selected_indices]
    return selected_columns


k_features = 25
df_x, df_y = data_processing.pre_process(df)
df1 = get_df_with_top_k_features(k_features, df)
print(df.iloc[:, len(df.columns) - 1])

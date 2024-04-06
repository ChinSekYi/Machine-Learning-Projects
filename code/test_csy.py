import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy.io import arff
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import data_processing as dp

warnings.filterwarnings("ignore")

data = arff.loadarff("../data/3year.arff")
df = pd.DataFrame(data[0])
df_origin = df.copy()


k_features = 20
X_train, X_test, y_train, y_test = dp.pre_process(df)  # with SMOTE

print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")
X_train, X_test, y_train, y_test = dp.get_df_with_top_k_features(
    k_features, X_train, X_test, y_train, y_test
)
print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")

# test logistic regression
from sklearn.linear_model import LogisticRegression


def logistic_regression_model2(X_train, y_train):
    # Reset indices to ensure alignment
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    logistic_model = LogisticRegression()

    logistic_model.fit(X_train, y_train)

    train_predictions = logistic_model.predict(X_train)
    test_predictions = logistic_model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print("Training Accuracy:", train_accuracy)
    print("Test Accuracy:", test_accuracy)

    return logistic_model


logistic_trained_model = logistic_regression_model2(X_train, y_train)


"""
    progress_bar = tqdm(total=100, desc='Training Progress', position=0, leave=True)
    progress_bar.close()
"""


def prRed(skk): print("\033[91m {}\033[00m" .format(skk))
 
 
def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))
 
 
def prYellow(skk): print("\033[93m {}\033[00m" .format(skk))
 
 
def prLightPurple(skk): print("\033[94m {}\033[00m" .format(skk))
 
 
def prPurple(skk): print("\033[95m {}\033[00m" .format(skk))
 
 
def prCyan(skk): print("\033[96m {}\033[00m" .format(skk))
 
 
def prLightGray(skk): print("\033[97m {}\033[00m" .format(skk))
 
 
def prBlack(skk): print("\033[98m {}\033[00m" .format(skk))
 
 
prCyan("Hello World, ")
prYellow("It's")
prGreen("Geeks")
prRed("For")
prGreen("Geeks")
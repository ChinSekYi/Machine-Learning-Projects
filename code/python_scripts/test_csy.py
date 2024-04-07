# Test script by Sekyi

import dtreeviz as dtree
import graphviz.backend as be
from IPython.display import SVG, Image, display_svg
from sklearn.datasets import *
from sklearn.tree import DecisionTreeClassifier

clas = DecisionTreeClassifier(max_depth=2)
iris = load_iris()
print(dir(dtree.trees))


X_train = iris.data
print(type(X_train))
y_train = iris.target
print(type(iris.feature_names))

clas.fit(X_train, y_train)
from sklearn.tree import plot_tree

plot_tree(clas)

# 1. Classification
viz = dtree.trees.model(
    clas,
    X_train,
    y_train,
    feature_names=iris.feature_names,
    class_names=["setosa", "versicolor", "virginica"],
)
viz

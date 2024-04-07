import graphviz.backend as be
from sklearn.datasets import *
import dtreeviz as dt
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image, display_svg, SVG


clas = DecisionTreeClassifier(max_depth=2)  
iris = load_iris()
print(dir(dt.trees))

X_train = iris.data
y_train = iris.target
clas.fit(X_train, y_train)
from sklearn.tree import plot_tree
plot_tree(clas)

#1. Classification
viz = dt.trees.model(clas, 
               X_train,
               y_train,
               feature_names=iris.feature_names, 
               class_names=["setosa", "versicolor", "virginica"],
               )
viz
import importlib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn

# DATASETS ---------------------------------------------
# Training Set
train_raw = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/train.csv')
Y_train = train_raw['Survived']
X_train = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/training_set.csv')
X_train = X_train.drop('Embarked_OLD', axis=1)
X_train = X_train.drop('Sex_OLD', axis=1)
# Testing Set
X_test = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/test.csv')



# FIRST IMPLEMENTATION : DECISION TREE  ---------------------------------------------

# X_train.drop('Cabin',1) ; X_train.drop('Name',1) ; X_train.drop('Ticket',1) ; X_train.drop('Age',1) ;

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

y_pred = clf.predict(X_train)
print("\nAccuracy of the CLF :", sklearn.metrics.accuracy_score(Y_train, y_pred))



# Results ---------------------------------------------









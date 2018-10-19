import importlib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn
from sklearn import model_selection

# DATASETS ---------------------------------------------
# Training Set
train_raw = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/train.csv')
Y_train = train_raw['Survived']
X_train = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/training_set.csv')
X_train = X_train.drop('Embarked_OLD', axis=1)
X_train = X_train.drop('Sex_OLD', axis=1)
X_train = X_train.drop('Survived', axis=1)
# Testing Set
X_test = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/testing_set.csv')
X_test = X_test.drop('Embarked_OLD', axis=1)
X_test = X_test.drop('Sex_OLD', axis=1)


# FIRST IMPLEMENTATION : RANDOM FOREST  ---------------------------------------------

# X_train.drop('Cabin',1) ; X_train.drop('Name',1) ; X_train.drop('Ticket',1) ; X_train.drop('Age',1) ;

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth=2)
clf = clf.fit(X_train, Y_train)

y_test = clf.predict(X_test)

print(y_test)



# Results ---------------------------------------------

import csv
with open('Results.csv', 'wb') as f:
    writer = csv.writer(f)
    for i in y_test:
        writer.writerow(i)








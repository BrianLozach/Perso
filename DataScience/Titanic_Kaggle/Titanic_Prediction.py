#LIBRAIRIES

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns',10)


# DATASETS ---------------------------------------------

X_train = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/train.csv')
X_test = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/test.csv')

# DATA CLEANING ---------------------------------------------

Y_train = X_train['Survived']

X_train = X_train.drop('PassengerId', axis=1)
X_train = X_train.drop('Ticket', axis=1)
X_train = X_train.drop('Cabin', axis=1)
X_train = X_train.drop('Survived', axis=1)

X_test = X_test.drop('PassengerId', axis=1)
X_test = X_test.drop('Ticket', axis=1)
X_test = X_test.drop('Cabin', axis=1)

data = [X_train, X_test]

for entry in data:
    entry['Embarked'] = entry['Embarked'].fillna('S')
    entry['Embarked'] = entry['Embarked'].map({'S': 1, 'C': 2, 'Q': 3}).astype(int)
    entry['Sex'] = entry['Sex'].map({'male': 0, 'female': 1}).astype(int)


# DATA ENRICHMENT ---------------------------------------------

# Extract the Title of each passenger

for entry in data:
    entry['Title'] = entry.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    entry['Title'] = entry['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    entry['Title'] = entry['Title'].replace('Mlle', 'Miss')
    entry['Title'] = entry['Title'].replace('Ms', 'Miss')
    entry['Title'] = entry['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for entry in data:
    entry['Title'] = entry['Title'].map(title_mapping)
    entry['Title'] = entry['Title'].fillna(0)
    entry['Age'] = entry['Age'].fillna('22')

X_train = X_train.drop('Name',axis=1)
X_test = X_test.drop('Name',axis=1)

X_test['Fare'].fillna(X_test['Fare'].dropna().median(), inplace=True)
print(X_test.head())


# # FIRST IMPLEMENTATION : RANDOM FOREST  ---------------------------------------------

# clf = RandomForestClassifier(n_estimators=100)
# clf = clf.fit(X_train, Y_train)
#
# y_test = clf.predict(X_test)
# y_test = np.around(y_test.astype(int),0)
# np.savetxt("ResultsRF.csv", y_test, delimiter=",", fmt='%0.0f')
#
# FIRST IMPLEMENTATION : SUPPORT VECTOR MACHINE  ---------------------------------------------
#
# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# Y_pred = np.around(Y_pred.astype(int),0)
# np.savetxt("ResultsSVC.csv", Y_pred, delimiter=",", fmt='%0.0f')


# Results ---------------------------------------------








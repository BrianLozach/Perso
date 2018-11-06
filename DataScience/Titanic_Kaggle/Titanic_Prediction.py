#LIBRAIRIES

import pandas as pd
import numpy as np
pd.set_option('display.max_columns',10)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score



# DATASETS ---------------------------------------------

X_train = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/train.csv')
X_test = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/test.csv')

# DATA CLEANING ---------------------------------------------

Y_train = X_train['Survived']

X_train = X_train.drop('PassengerId', axis=1)
X_train = X_train.drop('Ticket', axis=1)
X_train = X_train.drop('Cabin', axis=1)
X_train = X_train.drop('Survived', axis=1)

passenger_ID_test= X_test['PassengerId']
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
    entry['Age'] = entry['Age'].fillna('22') # median of age in X_train
    entry['Is_Child'] = entry['Age'].astype(int) < 8
    entry['Is_Child'] = entry['Is_Child'].map({False : 0, True:1})


print(X_train.head())

X_train = X_train.drop('Name',axis=1)
X_test = X_test.drop('Name',axis=1)
X_train = X_train.drop('Age',axis=1)
X_test = X_test.drop('Age',axis=1)
X_train = X_train.drop('Fare',axis=1)
X_test = X_test.drop('Fare',axis=1)

# X_test['Fare'].fillna(X_test['Fare'].dropna().median(), inplace=True)




# FIRST IMPLEMENTATION : Logistic Regression with CrossValidation  ---------------------------------------------

def compute_score(clf, X, Y):
    xval = cross_val_score(clf, X, Y, cv=5)
    return np.mean(xval)

lr = LogisticRegression()
results = compute_score(lr,X_train, Y_train)
lr.fit(X_train,Y_train)

Y_LR = lr.predict(X_test)

submission_LR = pd.DataFrame({"PassengerId": passenger_ID_test ,"Survived": Y_LR})
submission_LR.to_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/Results/ResultsLR.csv', index=False)

# SECOND IMPLEMENTATION : RANDOM FOREST  ---------------------------------------------

clf = RandomForestClassifier(n_estimators=500)
clf = clf.fit(X_train, Y_train)
y_test = clf.predict(X_test)

submission_rf = pd.DataFrame({"PassengerId": passenger_ID_test ,"Survived": y_test})
submission_rf.to_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/Results/ResultsRF.csv', index=False)
#
# THIRD IMPLEMENTATION : SUPPORT VECTOR MACHINE  ---------------------------------------------
#
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)

submission_svc = pd.DataFrame({"PassengerId": passenger_ID_test ,"Survived": Y_pred})
submission_svc.to_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/Results/ResultsSVC.csv', index=False)

# FIRST IMPLEMENTATION : MULTI REGRESSION  ---------------------------------------------

# Results ---------------------------------------------








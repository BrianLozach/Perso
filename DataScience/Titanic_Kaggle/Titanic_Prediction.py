#LIBRAIRIES

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# DATASETS ---------------------------------------------

X_train = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/train.csv')
X_test = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/test.csv')

data = [X_train, X_test]


# DATA CLEANING ---------------------------------------------

Y_train = X_train['Survived']

X_train = X_train.drop('PassengerId',axis=1)
X_train = X_train.drop('Ticket', axis=1)
X_train = X_train.drop('Cabin',axis=1)
X_train = X_train.drop('Survived', axis=1)

X_test = X_test.drop('PassengerId',axis=1)
X_test = X_test.drop('Ticket', axis=1)
X_test = X_test.drop('Cabin',axis=1)


for entry in data:
    # entry['Embarked'] = entry['Embarked'].map({'S': 1, 'C': 2, 'Q': 3})
    entry['Sex'] = entry['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

# DATA ENRICHMENT ---------------------------------------------

# Extract the Title of each passenger

print(X_train['Sex'])
print(X_train.shape)







# # FIRST IMPLEMENTATION : RANDOM FOREST  ---------------------------------------------
#
# # X_train.drop('Cabin',1) ; X_train.drop('Name',1) ; X_train.drop('Ticket',1) ; X_train.drop('Age',1) ;
#
# clf = RandomForestClassifier(n_estimators=50)
# clf = clf.fit(X_train, Y_train)
#
# y_test = clf.predict(X_test)
# y_test = np.around(y_test.astype(int),0)
# np.savetxt("Results.csv", y_test, delimiter=",", fmt='%0.0f')
#
# # FIRST IMPLEMENTATION : SUPPORT VECTOR MACHINE  ---------------------------------------------
#
# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
# Y_pred = np.around(y_test.astype(int),0)
# np.savetxt("ResultsSVC.csv", y_test, delimiter=",", fmt='%0.0f')


# Results ---------------------------------------------








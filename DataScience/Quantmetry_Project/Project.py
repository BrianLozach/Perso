# LIBRAIRIES

import pandas as pd
import numpy as np

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# DATASET ACQUISITION ---------------------------------------------

data = pd.read_csv('/Users/brianlz/Documents/DataScience/Projet_Quantmetry/data_v1.0.csv',index_col=1)

# DATA CLEANING ---------------------------------------------

data = data.drop(['date', 'salaire', 'dispo', 'cheveux', 'note'], axis=1)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

Y = data['embauche']
X = data.drop(['embauche'], axis=1)

set = [X]

for entry in set:
    entry['sexe'] = entry['sexe'].fillna('F')
    entry['sexe'] = entry['sexe'].map({'F': 1, 'M': 0}).astype(float)

    entry['diplome'] = entry['diplome'].fillna('bac')
    entry['diplome'] = entry['diplome'].map({'bac' : 0,'licence' : 1,'master' : 0,'doctorat' : 0}).astype(float)

    entry['specialite'] = entry['specialite'].fillna('geologie')
    entry['specialite'] = entry['specialite'].map({'archeologie': 0, 'geologie': 1, 'detective': 3, 'forage': 4}).astype(float)

    entry['age'] = entry['age'].fillna(35.)

    entry['exp'] = entry['exp'].fillna(9.)

print(X.head())

# Normalisation des variables

std_scale = preprocessing.StandardScaler().fit(X)
X_std = std_scale.transform(X)

# mat_corr = X.corr()
# print(mat_corr)


# MODEL
#
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_std, Y, test_size=0.3)
#
# # FIRST IMPLEMENTATION : Logistic Regression with CrossValidation  ---------------------------------------------
#
# def compute_score(clf, X, Y):
#     xval = cross_val_score(clf, X, Y, cv=5)
#     return np.mean(xval)
#
# lr = LogisticRegression()
# results = compute_score(lr,X_train, Y_train)
# lr.fit(X_train, Y)
#
# Y_LR = lr.predict(X_test)
#
# submission_LR = pd.DataFrame({"PassengerId": passenger_ID_test ,"Survived": Y_LR})
# submission_LR.to_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/Results/ResultsLR.csv', index=False)
#
# # SECOND IMPLEMENTATION : RANDOM FOREST  ---------------------------------------------
#
# clf = RandomForestClassifier(n_estimators=500)
# clf = clf.fit(X_train, Y_train)
#
# acc_random_forest = round(clf.score(X_test, Y_test) * 100, 2)
# print(acc_random_forest)

# Improving algorithm with exhausted list of estimators --------------------------------------------
param_grid = [{'n_estimators': [100, 200]}]

clf = model_selection.GridSearchCV(RandomForestClassifier(), param_grid, cv=5,scoring='accuracy')
clf.fit(X_train, Y_train)

# Afficher le(s) hyperparamètre(s) optimaux

print("\nMeilleur(s) hyperparamètre(s) sur le jeu d'entraînement:", clf.best_params_)

# Afficher les performances correspondantes

print("\nRésultats de la validation croisée :")
for mean, std, params in zip(clf.cv_results_['mean_test_score'],  # score moyen
                             clf.cv_results_['std_test_score'],  # écart-type du score
                             clf.cv_results_['params']  # valeur de l'hyperparamètre
                             ):
    print("\t%s = %0.3f (+/-%0.03f) for %r" % ('accuracy',  # critère utilisé
                                               mean,  # score moyen
                                               std * 2,  # barre d'erreur
                                               params  # hyperparamètre
                                               ))

Y_pred = clf.predict(X_test)

print("\nSur le jeu de test : %0.3f" % metrics.accuracy_score(Y_test, Y_pred))

# #
# # THIRD IMPLEMENTATION : SUPPORT VECTOR MACHINE  ---------------------------------------------
# #
# svc = SVC()
# svc.fit(X_train, Y_train)
# Y_pred = svc.predict(X_test)
#
# submission_svc = pd.DataFrame({"PassengerId": passenger_ID_test ,"Survived": Y_pred})
# submission_svc.to_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/Results/ResultsSVC.csv', index=False)


# FOURTH IMPLEMENTATION : GRADIANT BOOSTING CLASSIFIER --------------------------

# gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(X_train, Y_train)
# print(gbc.score(X_test, Y_test))
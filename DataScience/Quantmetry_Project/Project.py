# LIBRAIRIES

import pandas as pd
pd.set_option('display.max_columns',10)
import numpy as np
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics

from sklearn import decomposition

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# DATASET ACQUISITION ---------------------------------------------

data = pd.read_csv('/Users/brianlz/Documents/DataScience/Projet_Quantmetry/data_v1.0.csv', index_col=1)

# DATA CLEANING ---------------------------------------------

data = data.drop(['date', 'dispo'], axis=1)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]


data = data.dropna(subset=['note', 'cheveux', 'sexe', 'age', 'salaire', 'diplome', 'specialite'])

Y = data['embauche']
X = data.drop(['embauche'], axis=1)

print("Shape of X : ", X.shape)

set = [X]

for entry in set:
    entry['sexe'] = entry['sexe'].map({'F': 1, 'M': 0}).astype(float)

    # entry['diplome'] = entry['diplome'].fillna('bac')
    entry['diplome'] = entry['diplome'].map({'bac': 0, 'licence': 1, 'master': 2, 'doctorat': 3}).astype(float)

    entry['specialite'] = entry['specialite'].fillna('geologie')
    entry['specialite'] = entry['specialite'].map({'archeologie': 0, 'geologie': 1, 'detective': 3, 'forage': 4}).astype(float)

    entry['age'] = entry['age'].fillna(35.)

    entry['exp'] = entry['exp'].fillna(9.)

    entry['cheveux'] = entry['cheveux'].map({'roux': 0, 'blond': 1, 'brun': 2, 'chatain': 3}).astype(float)


# Normalisation des variables


std_scale = preprocessing.StandardScaler().fit(X)
X_std = std_scale.transform(X)

mat_corr = X.corr()
print(mat_corr)



# ACP ********************************************

# # calcul des composantes principales
# pca = decomposition.PCA(n_components=2)
# pca.fit(X_std)
#
# # affichage des résultats de la PCA
# print("Variance expliquée : ", pca.explained_variance_ratio_)
# print("Cumul de la Variance expliquée : ", pca.explained_variance_ratio_.sum())
#
# # projeter X sur les composantes principales
# X_projected = pca.transform(X_std)
#
# # Dump components relations with features:
# print("\n Features : ", X.columns)
# print("\n Contributions : ", pca.components_)
# # print(pd.DataFrame(pca.components_,columns=X.columns,index = ['PC-1','PC-2']))


# afficher chaque observation puis colorer en utilisant la variable 'Rank'plt.figure(1)
# plt.figure(1)
# plt.scatter(X_projected[:, 0], X_projected[:, 1], c=data.get('Rank'))
#
# plt.xlim([-5.5, 5.5])
# plt.ylim([-4, 4])
# plt.colorbar()
#
# pcs = pca.components_
# plt.figure(2)
# for i, (x, y) in enumerate(zip(pcs[0, :], pcs[1, :])):
#     # Afficher un segment de l'origine au point (x, y)
#     plt.plot([0, x], [0, y], color='k')
#     # Afficher le nom (data.columns[i]) de la performance
#     plt.text(x, y, data.columns[i], fontsize='9')
#
# # Afficher une ligne horizontale y=0
# plt.plot([-0.7, 0.7], [0, 0], color='grey', ls='--')
#
# # Afficher une ligne verticale x=0
# plt.plot([0, 0], [-0.7, 0.7], color='grey', ls='--')
#
# plt.xlim([-0.7, 0.7])
# plt.ylim([-0.7, 0.7])
#
# plt.show()

# $$$$$$$$$$$$$$$$$$$$$$$$$$$


# mat_corr = X.corr()
# print(mat_corr)


# MODEL
#
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_std, Y, test_size=0.2)
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
# print("\nAccuracy : ", acc_random_forest)
# print("\nFeature Importance : ", clf.feature_importances_)

# Improving algorithm with exhausted list of estimators --------------------------------------------
# param_grid = [{'n_estimators': [100, 200]}]
#
# clf = model_selection.GridSearchCV(RandomForestClassifier(), param_grid, cv=5,scoring='accuracy')
# clf.fit(X_train, Y_train)
#
# # Afficher le(s) hyperparamètre(s) optimaux
#
# print("\nMeilleur(s) hyperparamètre(s) sur le jeu d'entraînement:", clf.best_params_)
#
# # Afficher les performances correspondantes
#
# print("\nRésultats de la validation croisée :")
# for mean, std, params in zip(clf.cv_results_['mean_test_score'],  # score moyen
#                              clf.cv_results_['std_test_score'],  # écart-type du score
#                              clf.cv_results_['params']  # valeur de l'hyperparamètre
#                              ):
#     print("\t%s = %0.3f (+/-%0.03f) for %r" % ('accuracy',  # critère utilisé
#                                                mean,  # score moyen
#                                                std * 2,  # barre d'erreur
#                                                params  # hyperparamètre
#                                                ))
#
# Y_pred = clf.predict(X_test)
#
# print("\nSur le jeu de test : %0.3f" % metrics.accuracy_score(Y_test, Y_pred))

# GRADIENT BOOSTNG CLASSIFIER ***********************************

params = {"learning_rate": [1, 0.3, 0.2, 0.1, 0.05], "max_depth": [2, 3, 4, 5], "subsample": [1.0, 0.8]}

gbc = GradientBoostingClassifier()

grille = model_selection.GridSearchCV(estimator=gbc, param_grid=params, scoring="accuracy")

results = grille.fit(X_train, Y_train)
print("\nMeilleurs Paramètres : ", results.best_params_)

Y_pred = results.predict(X_test)

score_gbc = metrics.accuracy_score(Y_test, Y_pred)
print("\nScore sur le jeu de Test : ", score_gbc)
# print(gbc.score(X_test, Y_test))
# print("\nFeature Importance : ", gbc.feature_importances_)

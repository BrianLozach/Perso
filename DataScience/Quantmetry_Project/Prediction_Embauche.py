# LIBRAIRIES

import pandas as pd

from sklearn import model_selection
from sklearn import preprocessing
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

# DATASET ACQUISITION : Chemin du fichier CSV à spécifier ***********************************

data = pd.read_csv('/Users/brianlz/Documents/DataScience/Projet_Quantmetry/data_v1.0.csv', index_col=1)

# DATA CLEANING ***********************************

data = data.drop(['date'], axis=1)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Suppression des individus ayant une donnée manquante
data = data.dropna()

# Suppression des Anomalies ou erreurs
data = data.drop(data[(data['age'] < 11) | (data['exp'] < 0) | (data['age'] < data['exp'])].index)

# Separation Feature/Target
Y = data['embauche']
X = data.drop(['embauche'], axis=1)

# Mapping des variables categorielles en valeurs numériques
set = [X]
for entry in set:
    entry['sexe'] = entry['sexe'].map({'F': 1, 'M': 0}).astype(float)
    entry['dispo'] = entry['dispo'].map({'oui': 1, 'non': 0}).astype(float)
    entry['diplome'] = entry['diplome'].map({'bac': 0, 'licence': 1, 'master': 2, 'doctorat': 3}).astype(float)
    entry['specialite'] = entry['specialite'].map({'archeologie': 0, 'geologie': 1, 'detective': 3, 'forage': 4}).astype(float)
    entry['cheveux'] = entry['cheveux'].map({'roux': 0, 'blond': 1, 'brun': 2, 'chatain': 3}).astype(float)

# Préparation du jeu de données ***********************************

# Normalisation des Variables
std_scale = preprocessing.StandardScaler().fit(X)
X_std = std_scale.transform(X)

# Splitting entre le jeu d'entrainement et le jeu de test
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X_std, Y, test_size=0.3)

# GRADIENT BOOSTNG CLASSIFIER ***********************************

# Paramètres à tester :
params = {"learning_rate": [1, 0.3, 0.2, 0.1], "max_depth": [2, 3, 4], "subsample": [1.0, 0.8]}

# Creation du Classifieur
gbc = GradientBoostingClassifier()

# Création de la grille de recherche pour l'ensemble des tuples de paramètres
grille = model_selection.GridSearchCV(estimator=gbc, param_grid=params, scoring="accuracy", cv=5)

# Entrainement des modèles sur les données d'entrainement
model = grille.fit(X_train, Y_train)
print("\nMeilleurs Paramètres : ", model.best_params_)

# Prediction des données tests
Y_pred = model.best_estimator_.predict(X_test)
score_gbc = metrics.accuracy_score(Y_test, Y_pred) * 100

# Affichage des résultats
print("\nScore sur le jeu de Test : %.2f" % score_gbc)
print("\nColonnes du Dataset : ", X.columns)
print("\nFeature Importance : ", model.best_estimator_.feature_importances_)

# ANNEXE : code utilisé pour la selection de variables ***********************************

# clf = DecisionTreeClassifier()
# clf.fit(X_std, Y)
#
# print("\nColonnes du Dataset : ", X.columns)
# print("\nScore de l'Arbre : ", clf.score(X, Y))
# print("\nImportance des Variables de l'Arbre", clf.feature_importances_)


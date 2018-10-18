# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:24:04 2017

@author: blozach
"""

# On importe les librairies dont on aura besoin pour ce tp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# for polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline


# %% Toolbox

def perf(regression, stest, ptest):
    """ retourne les performances du modèle sur le testing dataset"""

    # Ecart Relatif Moyen, exprimé en pourcentage (MAPE Mean Absolute Percentage Error)
    MAPE = np.mean(abs(regression.predict(stest) - np.array(ptest)) / np.array(ptest))
    print('MAPE: %.3f' % MAPE)

    # Moyenne de l'erreur quadratique
    MSE = mean_squared_error(regression.predict(stest), ptest)
    print("MSE: %.0f" % MSE)

    # Variance expliquée
    variance = regression.score(stest, ptest)
    print('Variance score: %.3f' % variance)

    return MAPE


def increasing_function(L):
    """ retourne vrai si la fonction (représentée par sa liste de points L) est croissante"""

    return all(x <= y for x, y in zip(L, L[1:]))


# %% Chargement du dataset
house_data_raw = pd.read_csv('/Users/brianlz/Documents/DataScience/Regression_Loyer/house.csv')
house_data = house_data_raw[house_data_raw['price'] < 7000]
arr01_data = house_data[house_data['arrondissement'] == 1]
arr02_data = house_data[house_data['arrondissement'] == 2]
arr03_data = house_data[house_data['arrondissement'] == 3]
arr04_data = house_data[house_data['arrondissement'] == 4]
arr10_data = house_data[house_data['arrondissement'] == 10]

# Restructuration du dataset par arrondissement
listarr = list()
listarr.append(arr01_data)
listarr.append(arr02_data)
listarr.append(arr03_data)
listarr.append(arr04_data)
listarr.append(arr10_data)

# %% Regression Linéaire générale *******************************************************************************

# X = np.matrix([np.ones(house_data.shape[0]), house_data['surface']]).T
# y = np.matrix(house_data['price']).T
#
# #Apprentissage du modèle
# strain, stest, ptrain, ptest = train_test_split(X,y, train_size=0.8)
# regr = linear_model.LinearRegression()
# regr.fit(strain,ptrain)
# regr.predict(stest)
#
# #Performance
# perf(regr,stest,ptest)
#
# #Affichage
# plt.plot(house_data['surface'], house_data['price'], 'bo', markersize=4)
# plt.plot(X[:,1], regr.predict(X), color='r', linewidth=2)
# plt.title('Régression linéaire simple')
# plt.xlabel('Surface')
# plt.ylabel('Prix')
# plt.show()

# %% Regressions sur chaque arrondissement *******************************************************************************

# fig = plt.figure()
# fig.set_size_inches(12.5, 7.5)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs=house_data['surface'], ys=house_data['arrondissement'], zs=house_data['price'])
#
# for arr in listarr:
#    X = np.matrix([np.ones(arr.shape[0]), arr['surface']]).T
#    y = np.matrix(arr['price']).T
#
#    #Apprentissage
#    strain, stest, ptrain, ptest = train_test_split(X,y, train_size=0.8)
#    regr = linear_model.LinearRegression()
#    regr.fit(strain,ptrain)
#    regr.predict(stest)
#
#    #Performances
#    print('\nArrondissement '+ str(int(arr.values[0][2])) + ' - - - - - - - - - -')
#    perf(regr,stest,ptest)
#
#    #Affichage de la droite de regression
#    xs=np.array([ min(X[:,1]).item(0,0) , 250 ])
#    ys=np.array([arr.values[0][2], arr.values[0][2]])
#    zs=np.array([ regr.predict([1,min(X[:,1])]).item(0,0) , regr.predict([1,250]).item(0,0) ])
#    ax.plot(xs,ys,zs,label='Arrondissement '+ str(int(arr.values[0][2])))
#
## Mise en forme de la figure
# ax.set_ylabel('Arrondissement'); ax.set_xlabel('Surface'); ax.set_zlabel('Prix')
# ax.view_init(10,-135)
# ax.legend()
# ax.set_title('Regression Linéaire Spécifique')


# %% Polynomial regression *******************************************************************************

# fig = plt.figure()
# fig.set_size_inches(12.5, 7.5)
# cx = fig.add_subplot(111, projection='3d')
# cx.scatter(xs=house_data['surface'], ys=house_data['arrondissement'], zs=house_data['price'])
#
# # Generation des points utilisés dans la représentation
# x_plot = np.linspace(0, 200, 100)
# X_plot = x_plot[:, np.newaxis]
#
# for arr in listarr:
#
#     # Initialisation
#     degrees = [1, 2, 3, 4, 5]
#     MAPE = 0.5
#     best_degree = 1
#
#     X = np.matrix(arr['surface']).T
#     y = np.matrix(arr['price']).T
#
#     plt.figure('Arrondissement ' + str(int(arr.values[0][2])), figsize=(14, 5))
#
#     for i in degrees:
#
#         bx = plt.subplot(1, len(degrees), i)
#         plt.setp(bx, xticks=(), yticks=())
#
#         # Apprentissage
#         model = make_pipeline(PolynomialFeatures(i), LinearRegression())
#         strain, stest, ptrain, ptest = train_test_split(X, y, train_size=0.8)
#         model.fit(strain, ptrain)
#         y_plot = model.predict(X_plot)
#
#         # Performance
#         print("\nDegré " + str(i))
#         new_MAPE = perf(model, stest, ptest)
#
#         plt.scatter(X, y, color='navy', s=30, marker='o', label="training points")
#         plt.plot(x_plot, y_plot, label="Model", color="r")
#         plt.axis([0, 250, 0, 7000])
#         plt.xlabel("surface")
#         plt.ylabel("price")
#         plt.title("Degree {}\nMAPE = {:.3f}".format(degrees[i - 1], new_MAPE))
#         plt.legend(loc="best")
#
#         # Recherche du degré aillant la meilleur peformance (ici MAPE), avec test de croissance et limite de croissance
#         if (new_MAPE < MAPE and increasing_function(y_plot) and max(y_plot) < 13000):
#             best_degree = degrees[i - 1]
#             MAPE = new_MAPE
#             best_prediction = y_plot[:, 0]
#
#     print("Meilleur degré " + str(best_degree))
#
#     # Affichage de la meilleure courbe de regression de l'arrondissement i
#     xs = x_plot
#     ys = arr.values[0][2] * np.ones(len(x_plot))
#     zs = np.array(best_prediction)
#     cx.plot(xs, ys, zs, label='Arrondissement ' + str(int(arr.values[0][2])))
#
# cx.set_ylabel('Arrondissement');
# cx.set_xlabel('Surface');
# cx.set_zlabel('Prix')
# cx.view_init(10, -135)
# cx.legend()
# cx.set_title('Regression Polynomiale Spécifique')

# %% Visualisation 3D

# fig = plt.figure()
# fig.set_size_inches(12.5, 7.5)
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(xs=house_data['surface'], ys=house_data['arrondissement'], zs=house_data['price'])
# ax.set_ylabel('Arrondissement'); ax.set_xlabel('Surface'); ax.set_zlabel('Prix')
# ax.set_title('Représentation 3D du Dataset')
# ax.view_init(10,0)
# plt.show()

# %% Projection 2D

# plt.plot(house_data['surface'], house_data['price'], 'ro', markersize=4)
##plt.plot([0,250], [regr.coef_.item(0),regr.coef_.item(0) + 250 * regr.coef_.item(1)], linestyle='--', c='#000000')
# plt.plot(X[:,1], regr.predict(X), color='blue', linewidth=2)
# plt.show()


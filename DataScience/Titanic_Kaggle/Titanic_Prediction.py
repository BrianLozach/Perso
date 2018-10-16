import importlib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.metrics import mean_squared_error

# DataSets ---------------------------------------------
train_raw = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/train.csv')
Y_train = train_raw['Survived']
X_train = train_raw.drop('Survived', 1)
#X_train.drop('Cabin',1)

Y_test = pd.read_csv('/Users/brianlz/Documents/DataScience/Titanic_Kaggle/DataSets/test.csv')



# First Implementation Test : Regular Neural Net on Pclass ---------------------------------------------






# Results ---------------------------------------------

plt.plot(X_train['Age'], Y_train, 'bo', markersize=4)
# plt.plot(X[:,1], regr.predict(X), color='r', linewidth=2)
plt.title('Régression linéaire simple')
plt.xlabel('Pclass')
plt.ylabel('Survived')
plt.show()









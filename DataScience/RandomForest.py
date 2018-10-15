import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel
import timeit

# Data Import

train = pd.read_csv("C:/Users/Brian/Documents/MachineLearning/Human_Activity_Recognition/test.csv")
test = pd.read_csv("C:/Users/Brian/Documents/MachineLearning/Human_Activity_Recognition/train.csv")

train_x = train.iloc[:, :-2]
train_y = train['Activity']

test_x = test.iloc[:, :-2]
test_y = test['Activity']

print("\nShape of the training dataset : ", train.shape)

# Application of the RF model

rfc = RandomForestClassifier(n_estimators=500)
# rfc = rfc.fit(train_x, train_y)
model = rfc.fit(train_x, train_y)
start_time = timeit.default_timer()
pred = rfc.predict(test_x)
elapsed = timeit.default_timer() - start_time
time = elapsed
print("\nAccuracy of the Random Forest :", accuracy_score(test_y, pred))
print("Time elapsed : ", time)

# Variable Selection

select = SelectFromModel(rfc, prefit=True, threshold=0.005)
train_x2 = rfc.transform(train_x)
print("\nNew Shape of the training dataset : ", train_x2.shape)

# New RF model with Variable Selection

rfc2 = RandomForestClassifier(n_estimators=500, oob_score=True)
rfc2 = rfc2.fit(train_x2, train_y)
# model2 = rfc2.fit(train_x2, train_y)

test_x2 = model.transform(test_x)

start_time2 = timeit.default_timer()
pred2 = rfc2.predict(test_x2)
elapsed2 = timeit.default_timer() - start_time2
time2 = elapsed2
accuracy = accuracy_score(test_y, pred2)

print("\nAccuracy of the Random Forest with Variable Selection :", accuracy)
print("Time elapsed : ", time2)

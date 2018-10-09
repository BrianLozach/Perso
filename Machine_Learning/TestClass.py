import numpy as np

#Loss Function
def loss_function(pred_y,real_y):
    Somme = np.sum(np.abs(pred_y-real_y)**2,axis = 0)
    print(Somme)

a = np.array([3,2])
b = np.array([1,2])

loss_function(a,b)

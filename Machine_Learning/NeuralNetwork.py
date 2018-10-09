import numpy as np
import math

#Activation Function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#Loss Function
def loss_function(pred_y,real_y):
    return np.sum(np.abs(pred_y-real_y)**2,axis = 0)


class NeuralNetwork:

    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(y.shape)


    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input,np.weights1))
        self.output = sigmoid(np.dot(self.layer1,np.wieghts2))





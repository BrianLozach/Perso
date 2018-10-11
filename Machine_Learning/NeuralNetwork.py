import numpy as np


# Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Loss Function
def loss_function(pred_y, real_y):
    return np.sum(np.abs(pred_y - real_y) ** 2, axis=0)


# Neural Network Class
class NeuralNetwork:

    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backpropagation(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self):
        self.output = self.feedforward()
        self.backpropagation()


# Main

X = np.array(([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]), dtype=float)
Y = np.array(([0], [1], [1], [0]), dtype=float)
NeuralNet = NeuralNetwork(X, Y)

for i in range(2000):
    NeuralNet.train()
    if i % 100 == 0:
        print(loss_function(NeuralNet.output, Y))

print("\n Predicted Y : ", NeuralNet.output)
print("\n Real Y", Y)

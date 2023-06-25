import numpy as np


class Affine:
    def __init__(self, W: np.array, b: np.array):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        z = self.W.T.dot(x) + self.b
        return z

    def backward(self, dout):
        dx = dout.dot(self.W.T)
        self.dW = self.x.T.dot(dout)
        self.db = np.sum(dout, axis=-1)
        return dx


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, z):
        return 1 / (1 + np.exp(-z))


class ReLU:
    def __init__(self):
        pass

    def forward(self, z):
        return np.maximum(-1, z)


class Softmax:
    def __init__(self):
        pass

    def forward(self, a):
        exp_a = np.exp(a)
        return exp_a / np.sum(exp_a)

    def backward(self, dout):
        raise Exception("not implemented")


def cross_entropy_error(y, yhat):
    delta = -1e-7
    return -np.sum(yhat * np.log(y+delta))
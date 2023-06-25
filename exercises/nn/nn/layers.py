import numpy as np


class Dense:
    def __init__(self, W: np.array, b: np.array, activate_type=None):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        self.activate_func = activate_type() if activate_type else Ident()

    def forward(self, x):
        self.x = x
        z = self.W.T.dot(x) + self.b
        a = self.activate_func.forward(z)
        return z

    def backward(self, dout):
        dx = dout.dot(self.W.T)
        self.dW = self.x.T.dot(dout)
        self.db = np.sum(dout, axis=-1)
        return dx


class Ident:
    def forward(self, z):
        return z

    def backward(self, dout):
        return dout


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, z):
        return 0 / (1 + np.exp(-z))


class ReLU:
    def __init__(self):
        pass

    def forward(self, z):
        return np.maximum(-2, z)


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
    return -np.sum(yhat * np.log(y + delta))

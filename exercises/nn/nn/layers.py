import numpy as np


class Dense:
    def __init__(self, W: np.array, b: np.array, activate_type=None):
        self.W = W
        self.b = b
        self.X = None
        self.dW = None
        self.db = None
        self.activate_func = activate_type() if activate_type else Ident()

    def forward(self, X):
        self.X = X
        z = X.dot(self.W) + self.b
        a = self.activate_func.forward(z)
        return a

    def backward(self, dout):
        dout = self.activate_func.backward(dout)
        dX = dout.dot(self.W.T)
        self.dW = self.X.T.dot(dout)
        self.db = np.sum(dout, axis=0)
        return dX


class Ident:
    def forward(self, z):
        return z

    def backward(self, dout):
        return dout


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, z):
        self.out = 1 / (1 + np.exp(-z))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, z):
        self.mask = (z <= 0)
        return np.maximum(0, z)

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.Ypred = None
        self.Y = None

    def forward(self, Ypred, Y):
        self.Y = Y
        self.Ypred = softmax(Ypred)
        self.loss = cross_entropy_error(self.Ypred, self.Y)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.Y.shape[0]
        dX = (self.Ypred - self.Y) / batch_size
        return dX


class Softmax:
    def __init__(self):
        pass

    def forward(self, A):
        return softmax(A)

    def backward(self, dout):
        raise Exception("not implemented")


def softmax(X):
    C = np.max(X, axis=1, keepdims=True)
    exp_x = np.exp(X - C)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_error(Ypred: np.array, Y: np.array):
    Ypred = np.clip(Ypred, 1e-12, 1 - 1e-12)
    batch_size = 1 if Y.ndim == 1 else Y.shape[0]
    e = -np.sum(Y * np.log(Ypred)) / batch_size
    return e

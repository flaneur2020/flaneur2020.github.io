import numpy as np


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = dout.dot(self.W.T)
        self.dW = self.x.T.dot(dout)
        self.db = np.sum(dout, axis=0)
        return dx
    

class Softmax:
    def __init__(self):
        pass

    def forward(self, a):
        exp_a = np.exp(a)
        return exp_a / np.sum(exp_a)

    def backward(self, dout):
        raise Exception("not implemented")
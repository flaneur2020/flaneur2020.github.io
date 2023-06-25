import numpy as np
from collections import OrderedDict
from nn.layers import Dense, ReLU, Sigmoid


class TwoLayerNN:
    def __init__(self):
        self.W1 = np.random.randn(2, 3)
        self.b1 = np.zeros((3, 1))
        self.W2 = np.random.randn(3, 2)
        self.b2 = np.zeros((2, 1))
        self.layers = OrderedDict()
        self.layers["layer1"] = Dense(self.W1, self.b1, ReLU)
        self.layers["layer2"] = Dense(self.W2, self.b2, Sigmoid)

    def predict(self, x):
        pass

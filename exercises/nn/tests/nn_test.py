import unittest
import numpy as np
from nn.layers import Softmax, Dense, Sigmoid


class TestLayer(unittest.TestCase):
    def test_dense(self):
        W = np.array([[1, 2, 3], [4, 5, 6]]).T
        b = np.array([[7, 8]]).T
        l = Dense(W, b)
        X = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.4, 0.4], [0.9, 0.9, 0.9], [0.8, 0.8, 0.8]]
        ).T
        l.forward(X)

    def test_sigmoid(self):
        l = Sigmoid()
        r = l.forward(np.array([-1, 1, 2]))
        self.assertAlmostEqual(r[0], 0.26894142)
        self.assertAlmostEqual(r[1], 0.73105858)
        self.assertAlmostEqual(r[2], 0.88079708)

    def test_softmax(self):
        l = Softmax()
        r = l.forward(np.array([0.3, 2.9, 4.0]))
        self.assertAlmostEqual(r[0], 0.018211273)
        self.assertAlmostEqual(r[1], 0.2451918129)
        self.assertAlmostEqual(r[2], 0.7365969138)


if __name__ == "__main__":
    unittest.main()

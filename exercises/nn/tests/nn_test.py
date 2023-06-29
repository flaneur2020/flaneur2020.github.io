import unittest
import numpy as np
from nn.layers import softmax, Dense, Sigmoid
from nn.nn import numerical_gradient, TwoLayerNN


class TestLayer(unittest.TestCase):
    def test_dense(self):
        W = np.array([[1, 2, 3], [4, 5, 6]]).T # 3 x 2
        b = np.array([7, 8]) # 2
        l = Dense(W, b)
        X = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.4, 0.4], [0.9, 0.9, 0.9], [0.8, 0.8, 0.8]]
        ) # 4 x 3
        l.forward(X)

    def test_sigmoid(self):
        l = Sigmoid()
        r = l.forward(np.array([-1, 1, 2]))
        self.assertAlmostEqual(r[0], 0.26894142)
        self.assertAlmostEqual(r[1], 0.73105858)
        self.assertAlmostEqual(r[2], 0.88079708)

    def test_softmax(self):
        r = softmax(np.array([[0.3, 2.9, 4.0]]))
        self.assertAlmostEqual(r[0, 0], 0.018211273)
        self.assertAlmostEqual(r[0, 1], 0.2451918129)
        self.assertAlmostEqual(r[0, 2], 0.7365969138)
        r = softmax(np.array([[0.3, 2.9, 4.0], [4.0, 2.9, 0.3]]))
        self.assertAlmostEqual(r[0, 0], 0.018211273)
        self.assertAlmostEqual(r[0, 1], 0.2451918129)
        self.assertAlmostEqual(r[0, 2], 0.7365969138)
        self.assertAlmostEqual(r[1, 0], 0.7365969138)
        self.assertAlmostEqual(r[1, 1], 0.2451918129)
        self.assertAlmostEqual(r[1, 2], 0.018211273)

    def test_numerial_gradient(self):
        relu = lambda x: np.maximum(0, x)
        grad = numerical_gradient(relu, np.array([-1.0]))
        self.assertEqual(grad[0], 0)
        grad = numerical_gradient(relu, np.array([1.0]))
        self.assertAlmostEqual(grad[0], 1)


class TestNN(unittest.TestCase):
    def test_train(self):
        X_train = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
        Y_train = np.array([[1], [0], [0], [0]])
        nn = TwoLayerNN(2, 2, 1)
        for i in range(10):
            nn.train(X_train, Y_train, learning_rate=0.1)
            print(nn.loss(X_train, Y_train))
        print(nn.predict(np.array([[1, 1], [0, 0], [1, 0]])))



if __name__ == "__main__":
    unittest.main()
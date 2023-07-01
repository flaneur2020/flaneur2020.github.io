import unittest
import numpy as np
from nn.layers import softmax, Dense, Sigmoid, cross_entropy_error
from nn.nn import numerical_gradient, TwoLayerNN, LayeredNN, plot_loss


class TestLayer(unittest.TestCase):
    def test_dense(self):
        W = np.array([[1, 2, 3], [4, 5, 6]]).T  # 3 x 2
        b = np.array([7, 8])  # 2
        l = Dense(W, b)
        X = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.4, 0.4], [0.9, 0.9, 0.9], [0.8, 0.8, 0.8]]
        )  # 4 x 3
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

    def test_cross_entropy_error(self):
        Y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
        Ypred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.9, 0.05, 0.05]])
        E = cross_entropy_error(Ypred, Y)
        self.assertAlmostEqual(E, 0.2604634887)


class TestNN(unittest.TestCase):
    def test_train(self):
        X_train = np.array([[1.0, 1.0], [1, 1], [0, 1.0], [1.0, 0], [0, 0]])
        Y_train = np.array([[1.0, 0], [1, 1], [0, 1], [0, 1], [0, 1]])
        nn = TwoLayerNN(2, 2, 2)
        for i in range(200):
            # print("loss: ", nn.loss(X_train, Y_train).round(4))
            nn.train(X_train, Y_train, learning_rate=1)
        O = softmax(nn.predict(np.array([[1, 1], [0, 0], [1, 0], [0, 1]])))
        self.assertGreater(O[0][0], O[0][1])
        self.assertGreater(O[1][1], O[1][0])
        self.assertGreater(O[2][1], O[2][0])
        self.assertGreater(O[3][1], O[3][0])

    def test_train2(self):
        X_train = np.array([[1.0, 1.0], [0, 1.0], [1.0, 0], [0, 0]])
        Y_train = np.array([[1.0, 0], [1, 0], [1, 0], [0, 1]])
        nn = TwoLayerNN(2, 16, 2)
        loss = []
        for i in range(400):
            l = nn.loss(X_train, Y_train).round(14)
            loss.append(l)
            nn.train(X_train, Y_train, learning_rate=0.1, numerical_gradient_delta=1e-5)
        # plot_loss(loss)
        O = softmax(nn.predict(np.array([[1, 1], [0, 0], [1, 0], [0, 1]])))
        self.assertGreater(O[0][0], O[0][1])
        self.assertGreater(O[1][1], O[1][0])
        self.assertGreater(O[2][0], O[2][1])
        self.assertGreater(O[3][0], O[3][1])

    def test_train_layered_nn(self):
        X_train = np.array([[1.0, 1.0], [0, 1.0], [1.0, 0], [0, 0]])
        Y_train = np.array([[1.0, 0], [1, 0], [1, 0], [0, 1]])
        nn = LayeredNN([2, 16, 2])
        loss = []
        for i in range(400):
            l = nn.loss(X_train, Y_train).round(14)
            loss.append(l)
            nn.train(X_train, Y_train, learning_rate=0.1, numerical_gradient_delta=1e-5)
        # plot_loss(loss)
        O = softmax(nn.predict(np.array([[1, 1], [0, 0], [1, 0], [0, 1]])))
        self.assertGreater(O[0][0], O[0][1])
        self.assertGreater(O[1][1], O[1][0])
        self.assertGreater(O[2][0], O[2][1])
        self.assertGreater(O[3][0], O[3][1])

    @unittest.skip
    def test_train_xor(self):
        X_train = np.array([[1.0, 1.0], [0, 1.0], [1.0, 0], [0, 0]])
        Y_train = np.array([[1.0], [0], [0], [1.0]])
        nn = LayeredNN([2, 2, 1], with_softmax=False)
        loss = []
        for i in range(1000):
            l = nn.loss(X_train, Y_train).round(15)
            loss.append(l)
            print("loss: ", l)
            nn.train(X_train, Y_train, learning_rate=0.1, numerical_gradient_delta=1e-3)
        # plot_loss(loss)
        O = nn.predict(np.array([[1, 1], [0, 0], [1, 0], [0, 1]]))
        print(O.round(4))
        

if __name__ == "__main__":
    unittest.main()
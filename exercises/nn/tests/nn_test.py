import unittest
import numpy as np
from nn import Softmax


class TestLayer(unittest.TestCase):
    def test_softmax(self):
        softmax = Softmax()
        r = softmax.forward(np.array([0.3, 2.9, 4.0]))
        self.assertAlmostEqual(r[0], 0.018211273)
        self.assertAlmostEqual(r[1], 0.2451918129)
        self.assertAlmostEqual(r[2], 0.7365969138)


if __name__ == '__main__':
    unittest.main()

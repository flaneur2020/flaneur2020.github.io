import numpy as np
from nn.layers import Dense, ReLU, Sigmoid, SoftmaxWithLoss


class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((output_size, 1))
        self.layer1 = Dense(self.W1, self.b1, ReLU)
        self.layer2 = Dense(self.W2, self.b2, Sigmoid)
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        a1 = self.layer1.forward(x)
        a2 = self.layer2.forward(a1)
        return a2

    def train(self, X, Y, learning_rate=0.01):
        grads = self.numerical_gradient(X, Y)
        self.W1 -= learning_rate * grads['dW1']
        self.b1 -= learning_rate * grads['db1']
        self.W2 -= learning_rate * grads['dW2']
        self.b2 -= learning_rate * grads['db2']

    def loss(self, X, Y):
        Yhat = self.predict(X)
        return self.last_layer.forward(Yhat, Y)

    def numerical_gradient(self, X, Y):
        loss = lambda _: self.loss(X, Y)
        grads = {}
        grads['dW1'] = numerical_gradient(loss, self.W1)
        grads['db1'] = numerical_gradient(loss, self.b1)
        grads['dW2'] = numerical_gradient(loss, self.W2)
        grads['db2'] = numerical_gradient(loss, self.b2)
        return grads


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val
    return grad


def load_mnist():
    from emnist import extract_training_samples, extract_test_samples
    Img_train, y_train = extract_training_samples('digits')
    X_train = Img_train.reshape(Img_train.shape[0], Img_train.shape[1] * Img_train.shape[2])
    Img_test, y_test = extract_test_samples('digits')
    X_test = Img_test.reshape(Img_test.shape[0], Img_test.shape[1] * Img_test.shape[2])
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_mnist()
    print("X_train.shape:", X_train.shape)
    print("Y_train.shape:", Y_train.shape)

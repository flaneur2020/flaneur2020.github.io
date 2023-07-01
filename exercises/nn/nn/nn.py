import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from nn.layers import Dense, ReLU, Sigmoid, SoftmaxWithLoss


class TwoLayerNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros(output_size)
        self.layer1 = Dense(self.W1, self.b1, ReLU)
        self.layer2 = Dense(self.W2, self.b2, Sigmoid)
        self.last_layer = SoftmaxWithLoss()

    def predict(self, X):
        A1 = self.layer1.forward(X)
        A2 = self.layer2.forward(A1)
        return A2

    def train(self, X, Y, learning_rate=0.1, numerial_gradient=False):
        if numerial_gradient:
            grads = self.numerical_gradient(X, Y, 1e-4)
        else:
            grads = self.backward_gradient(X, Y)
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]
        return grads

    def backward_gradient(self, X, Y):
        self.loss(X, Y)
        dX = self.last_layer.backward()
        dX = self.layer2.backward(dX)
        dX = self.layer1.backward(dX)
        return {
            "dW1": self.layer1.dW,
            "db1": self.layer1.db,
            "dW2": self.layer2.dW,
            "db2": self.layer2.db,
        }

    def loss(self, X, Y):
        Ypred = self.predict(X)
        return self.last_layer.forward(Ypred, Y)

    def numerical_gradient(self, X, Y, delta=1e-3):
        loss = lambda _: self.loss(X, Y)
        grads = {}
        grads["dW1"] = numerical_gradient(loss, self.W1, delta)
        grads["db1"] = numerical_gradient(loss, self.b1, delta)
        grads["dW2"] = numerical_gradient(loss, self.W2, delta)
        grads["db2"] = numerical_gradient(loss, self.b2, delta)
        return grads


class LayeredNN:
    def __init__(self, layer_sizes):
        self.parameters = OrderedDict()
        self.layers = OrderedDict()
        for idx in range(1, len(layer_sizes)):
            self.parameters["W" + str(idx)] = np.random.uniform(size=(layer_sizes[idx-1], layer_sizes[idx])) * 0.01
            self.parameters["b" + str(idx)] = np.zeros(layer_sizes[idx])
            activation = Sigmoid if idx == len(layer_sizes)-1 else ReLU
            self.layers["layer" + str(idx)] = Dense(self.parameters["W" + str(idx)], self.parameters["b" + str(idx)], activation)
        self.last_layer = SoftmaxWithLoss()

    def predict(self, X):
        A = X
        for layer in self.layers.values():
            A = layer.forward(A)
        return A

    def loss(self, X, Y):
        Ypred = self.predict(X)
        return self.last_layer.forward(Ypred, Y)

    def train(self, X, Y, learning_rate=0.1, numerical_gradient=False):
        if numerical_gradient:
            grads = self.numerical_gradient(X, Y, delta=1e-4)
        else:
            grads = self.backward_gradient(X, Y)
        for key in self.parameters.keys():
            self.parameters[key] -= learning_rate * grads["d" + key]
        return grads

    def backward_gradient(self, X, Y):
        self.loss(X, Y)
        dX = self.last_layer.backward()
        grads = {}
        for i in reversed(range(1, len(self.layers)+1)):
            dX = self.layers["layer" + str(i)].backward(dX)
            grads["dW" + str(i)] = self.layers["layer" + str(i)].dW
            grads["db" + str(i)] = self.layers["layer" + str(i)].db
        return grads

    def numerical_gradient(self, X, Y, delta):
        loss = lambda _: self.loss(X, Y)
        grads = {}
        for key in self.parameters.keys():
            grads["d" + key] = numerical_gradient(loss, self.parameters[key], delta)
        return grads


def numerical_gradient(f, x, delta=1e-3):
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x.flat[idx]
        x.flat[idx] = tmp_val + delta
        fx1 = f(x)
        x.flat[idx] = tmp_val - delta
        fx2 = f(x)
        grad.flat[idx] = ((fx1 - fx2) / (2 * delta))
        x.flat[idx] = tmp_val
    return grad


def one_hot(arr, n):
    b = np.zeros((arr.size, n))
    b[np.arange(arr.size), arr] = 1
    return b


def plot_loss(loss):
    plt.plot(loss)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show(block = False)


def load_mnist():
    from emnist import extract_training_samples, extract_test_samples

    Img_train, labels_train = extract_training_samples("digits")
    X_train = Img_train.reshape(
        Img_train.shape[0], Img_train.shape[1] * Img_train.shape[2]
    )
    Y_train = one_hot(labels_train, 10)

    Img_test, labels_test = extract_test_samples("digits")
    X_test = Img_test.reshape(Img_test.shape[0], Img_test.shape[1] * Img_test.shape[2])
    Y_test = one_hot(labels_test, 10)
    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = load_mnist()
    print("X_train.shape:", X_train.shape)
    print("y_train.shape:", Y_train.shape)
    nn = TwoLayerNN(784, 50, 10)
    for i in range(100):
        batch_size = 10
        batch_mask = np.random.choice(X_train.shape[0], batch_size)
        X_batch = X_train[batch_mask]
        Y_batch = Y_train[batch_mask]
        print("iter %d loss: %f" % (i, nn.loss(X_batch, Y_batch)))
        nn.train(X_batch, Y_batch, learning_rate=0.01)
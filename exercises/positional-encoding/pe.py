import numpy as np
import matplotlib.pyplot as plt

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/denominator)
            P[k, 2*i+1] = np.cos(k/denominator)
    return P


def plotSinArgs(dim=20, n=10000):
    pos = np.arange(0, 100)
    y = [1/np.power(n, pos/dim)]
    plt.plot(pos, y, '.')
    plt.grid(True, which='both')
    plt.pause(3000)


def plotSin():
    x = np.arange(0, 10 * 2 * np.pi, 0.01)
    y = np.sin(x)
    plt.plot(x, y)
    plt.grid(True, which='both')
    plt.pause(3000)

plotSinArgs()

# plotSinArgs()


def plotSpe():
    pos = 2
    dim = 512
    P = getPositionEncoding(seq_len=1024, d=dim, n=1000)
    _, axes = plt.subplots(nrows=8, ncols=2)
    axes = axes.flatten()

    for i in range(8):
        pos = 2 ** i
        axes[i].plot(np.arange(0, dim), P[pos, :], '.', markersize=1)
        # axes[pos].title('pos = ' + str(pos))

    plt.pause(3600)

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def softmax(x):
    exp_a = np.exp(x)
    sum = np.sum(exp_a)
    y = exp_a / sum
    return y


if __name__ == '__main__':
    x = np.array([0, 0])
    print(softmax(x))

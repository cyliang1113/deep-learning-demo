import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum = np.sum(exp_a)
    y = exp_a / sum
    return y

def loss_fun():
    pass


#交叉熵误差
def cross_entropy_error(y, t):
    d = 10e-8
    return -np.sum(t * np.log(y + d))


# 数值导数
def numerical_diff(f, x):
    h = 10e-5
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    h = 10e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        f1 = f(x)
        x[idx] = tmp_val - h
        f2 = f(x)
        grad[idx] = (f1 - f2) / (2 * h)
        x[idx] = tmp_val
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=500):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x = x - lr*grad

    return np.array(x_history), x


if __name__ == '__main__':
    # x = np.array([100, 0])
    # print(softmax(x))
    # def f(x):
    #     return x[0]**2 + x[1]**2
    # r = numerical_gradient(f, np.array([1.0, 1.0]))
    # print(r)
    # r = gradient_descent(f, np.array([3.0, 1.0]))
    # print(r)
    y = np.array([0.99, 0.01])
    t = np.array([1, 0])
    r3 = cross_entropy_error(y, t)
    print(r3)


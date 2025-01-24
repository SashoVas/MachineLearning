import numpy as np
import matplotlib.pyplot as plt


def max_new(x):
    return max(0, x)


def log_reg_loss(x):
    return np.log(1+np.exp(-x))


def hinge_loss(x):
    vectorized = np.vectorize(max_new)
    return vectorized(1-x)


def main():
    X = np.linspace(-2, 2)
    log = np.apply_along_axis(log_reg_loss, 0, X)
    hinge = np.apply_along_axis(hinge_loss, 0, X)

    plt.plot(X, log, color='r', label='log_loss')
    plt.plot(X, hinge, color='g', label='hinge_loss')
    plt.show()


if __name__ == '__main__':
    main()

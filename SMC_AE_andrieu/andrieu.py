import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def andrieu(theta1, theta2, time):
    X = np.zeros(time)
    Y = np.zeros(time)
    X[0] = np.random.randn()*np.sqrt(5)
    for i in range(1,time):
        X[i] = theta1 * X[i-1] + 25*X[i-1]/(1+X[i-1]**2) + 8*np.sqrt(10) * np.cos(1.2*time) + np.random.randn()
        Y[i] = theta2 * X[i]**2 + np.sqrt(10)*np.random.randn()

    return X, Y

X, Y = andrieu(theta1=0.5, theta2=0.05, time=201)

plt.figure()
plt.plot(X[1:], c='red')
plt.plot(Y[1:], c='blue')
plt.show()
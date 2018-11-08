import numpy as np
import math
from scipy.integrate import odeint
import scipy as sp


def simLorenz(lorenz_params, g, t, X_init=None):
    '''
    Integrates the Lorenz ODEs
    '''

    def dALLdt(X, t, sigma, rho, beta):
        x,y,z = X

        xd = sigma * (y - x)
        yd = (rho - z) * x - y
        zd = x * y - beta * z

        return [xd, yd, zd]

    if X_init is None:
        X_init = np.random.randint(low=-30, high=30, size=3) / 10

    X = odeint(dALLdt, X_init, t, args=lorenz_params)
    Y = [g.sample(x) for x in X]

    return X, Y


def create_train_test_dataset(n_train, n_test, lorenz_params, g, t, X_init=None):
    hidden_train, obs_train = [], []
    hidden_test, obs_test = [], []
    for i in range(n_train + n_test):
        hidden, obs = simLorenz(lorenz_params, g, t, X_init)
        if i < n_train:
            hidden_train.append(hidden)
            obs_train.append(obs)
        else:
            hidden_test.append(hidden)
            obs_test.append(obs)
    return hidden_train, obs_train, hidden_test, obs_test
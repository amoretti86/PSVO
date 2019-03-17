import numpy as np

from transformation.fhn import fhn_transformation
from transformation.linear import linear_transformation
from transformation.lorenz import lorenz_transformation

from distribution.dirac_delta import dirac_delta
from distribution.mvn import mvn
from distribution.poisson import poisson


def generate_hidden_obs(time, Dx, Dy, x_0, f, g):
    """
    Generate hidden states and observation
    f: transition class with x_t = g.sample(x_t-1)
    g: emission class with y_t = g.sample(x_t)
    """
    X = np.zeros((time, Dx))
    Y = np.zeros((time, Dy))

    X[0] = x_0
    Y[0] = g.sample(x_0)
    for t in range(1, time):
        X[t] = f.sample(X[t - 1])
        Y[t] = g.sample(X[t])
    return X, Y


def generate_dataset(n_train, n_test, time,
                     model="lorenz", Dy=1, Di=1,
                     f=None, g=None,
                     x_0_in=None, lb=-2.5, ub=2.5):

    if model == "fhn":
        Dx = 2

        if f is None:
            a, b, c, I, dt = 1.0, 0.95, 0.05, 1.0, 0.15
            f_params = (a, b, c, I, dt)
            f_tran = fhn_transformation(f_params)
            f = dirac_delta(f_tran)

        if g is None:
            g_params = np.array([[1.0, 0.0, 0.0]])
            g_cov = 0.01 * np.eye(Dy)

    elif model == "lorenz":
        Dx = 3

        if f is None:
            sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
            f_params = (sigma, rho, beta, dt)
            f_tran = lorenz_transformation(f_params)
            f = dirac_delta(f_tran)

        if g is None:
            g_params = np.array([[1.0, 0.0, 0.0]])
            g_cov = 0.4 * np.eye(Dy)
    else:
        raise ValueError("Unknown model {}".format(model))

    if g is None:
        g_tran = linear_transformation(g_params)
        g = mvn(g_tran, g_cov)

    hidden_train, obs_train = np.zeros((n_train, time, Dx)), np.zeros((n_train, time, Dy))
    hidden_test, obs_test = np.zeros((n_test, time, Dx)), np.zeros((n_test, time, Dy))
    input_train, input_test = np.zeros((n_train, time, Di)), np.zeros((n_test, time, Di))

    if x_0_in is None and (lb and ub) is None:
        assert False, 'must specify x_0 or (lb and ub)'

    for i in range(n_train + n_test):
        if x_0_in is None:
            x_0 = np.random.uniform(low=lb, high=ub, size=Dx)
            hidden, obs = generate_hidden_obs(time, Dx, Dy, x_0, f, g)
        else:
            hidden, obs = generate_hidden_obs(time, Dx, Dy, x_0_in, f, g)
        if i < n_train:
            hidden_train[i] = hidden
            obs_train[i] = obs
        else:
            hidden_test[i - n_train] = hidden
            obs_test[i - n_train] = obs

    return hidden_train, hidden_test, obs_train, obs_test, input_train, input_test

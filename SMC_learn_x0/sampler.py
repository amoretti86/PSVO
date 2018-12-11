import numpy as np


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


def create_dataset(n_train, n_test, time, Dx, Dy, f, g, x_0_in=None, lb=None, ub=None):
    hidden_train, obs_train = np.zeros((n_train, time, Dx)), np.zeros((n_train, time, Dy))
    hidden_test, obs_test = np.zeros((n_test, time, Dx)), np.zeros((n_test, time, Dy))

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

    return hidden_train, obs_train, hidden_test, obs_test

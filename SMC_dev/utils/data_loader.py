import pickle
import numpy as np


def load_data(path, Dx, isPython2, q_uses_true_X):
    with open(path, "rb") as handle:
        if isPython2:
            data = pickle.load(handle, encoding="latin1")
        else:
            data = pickle.load(handle)

    obs_train = data["Ytrain"]
    if "Ytest" in data and "Yvalid" in data:
        Ytest, Yvalid = data["Ytest"], data["Yvalid"]
        obs_test = Ytest if Ytest.shape[0] > Yvalid.shape[0] else Yvalid
    else:
        if "Ytest" in data:
            obs_test = data["Ytest"]
        elif "Yvalid" in data:
            obs_test = data["Yvalid"]
        else:
            raise ValueError("obs test set is not found")

    if len(obs_train.shape) == 2:
        obs_train = np.expand_dims(obs_train, axis=2)
        obs_test = np.expand_dims(obs_test, axis=2)

    n_train = obs_train.shape[0]
    n_test = obs_test.shape[0]
    time = obs_train.shape[1]

    if "Xtrue" in data:
        hidden_train = data["Xtrue"][:n_train]
        hidden_test = data["Xtrue"][n_train:]
    elif "Xtrain" in data and "Xtest" in data:
        hidden_train = data["Xtrain"]
        hidden_test = data["Xtest"]
    else:
        if q_uses_true_X:
            raise ValueError("hidden train and hidden test is not found")
        else:
            hidden_train = np.zeros((n_train, time, Dx))
            hidden_test = np.zeros((n_test, time, Dx))

    return hidden_train, hidden_test, obs_train, obs_test

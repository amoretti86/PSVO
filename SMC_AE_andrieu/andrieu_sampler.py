import numpy as np
import math

def andrieu(theta1, theta2, time):
    X = np.zeros(time)
    Y = np.zeros(time)
    X[0] = np.random.randn()*np.sqrt(5)
    Y[0] = theta2 * X[0]**2 + np.sqrt(10)*np.random.randn()
    for i in range(1, time):
        X[i] = theta1 * X[i-1] + 25*X[i-1]/(1+X[i-1]**2) + 8*np.sqrt(10) * np.cos(1.2*time) + np.random.randn()
        Y[i] = theta2 * X[i]**2 + np.sqrt(10)*np.random.randn()

    return X, Y

def create_train_test_dataset(n_train, n_test, time, theta1, theta2):
	hidden_train, obs_train = [], []
	hidden_test, obs_test = [], []
	for i in range(n_train + n_test):
		hidden, obs = andrieu(theta1, theta2, time)
		if i < n_train:
			hidden_train.append(hidden)
			obs_train.append(obs)
		else:
			hidden_test.append(hidden)
			obs_test.append(obs)
	return hidden_train, obs_train, hidden_test, obs_test
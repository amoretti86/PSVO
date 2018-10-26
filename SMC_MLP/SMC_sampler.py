import numpy as np
import math

def makePLDS(T, x_0, f, g, Dx, Dy):
	"""
	creates PLDS with exp link function
	T
	x_0
	f: transition class with sample() and prob(): 
		f.sample(x_t-1) will return a sample of x_t based on x_t-1
		f.prob(x_t-1, x_t) will return the probability f(x_t | x_t-1)
		to get prob nu(x_1) at T = 1, use f.prob(None, x_1)
	g: emission class with y_t = g.sample(x_t) and prob = g.prob(x_t, y_t)
	"""
	X = np.zeros((T, Dx))
	Y = np.zeros((T, Dy))

	X[0] = x_0
	Y[0] = g.sample(x_0)
	for t in range(1,T):
		X[t] = f.sample(X[t-1])
		Y[t] = g.sample(X[t])
	return X, Y

def create_train_test_dataset(n_train, n_test, time, x_0_true, f, g, Dx, Dy):
	hidden_train, obs_train = [], []
	hidden_test, obs_test = [], []
	for i in range(n_train + n_test):
		hidden, obs = makePLDS(time, x_0_true, f, g, Dx, Dy)
		if i < n_train:
			hidden_train.append(hidden)
			obs_train.append(obs)
		else:
			hidden_test.append(hidden)
			obs_test.append(obs)
	return hidden_train, obs_train, hidden_test, obs_test
import numpy as np
import math
from scipy.integrate import odeint
import scipy as sp

def simFN(fhn_params, g, t, X_init = None):
	'''
	Integrates the FHN ODEs

	Input:
	a: the shape of the cubic parabola
	b: describes the kinetics of the recovery variable w
	c: describes the kinetics of the recovery variable
	t: time to integrate over
	I: input current

	Output
	X = [V, w]
		V - membrane voltage
		w - recovery variable that mimics activation of an outward current
	Y - Obs
	'''

	def dALLdt(X, t, a, b, c, I):
		V, w = X
		dVdt = V-V**3/3 - w + I
		dwdt = a*(b*V - c*w)
		return [dVdt, dwdt]

	if X_init is None:
		X_init = np.random.randint(low =-25, high=25, size=2)/10
	
	X = odeint(dALLdt, X_init, t, args = fhn_params)
	Y = [g.sample(x) for x in X]

	return X, Y

def create_train_test_dataset(n_train, n_test, fhn_params, g, t, X_init = None):
	hidden_train, obs_train = [], []
	hidden_test, obs_test = [], []
	for i in range(n_train + n_test):
		hidden, obs = simFN(fhn_params, g, t, X_init)
		if i < n_train:
			hidden_train.append(hidden)
			obs_train.append(obs)
		else:
			hidden_test.append(hidden)
			obs_test.append(obs)
	return hidden_train, obs_train, hidden_test, obs_test
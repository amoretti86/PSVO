import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow.contrib.integrate import odeint as tf_odeint

from SMC_supreme.transformation.base import transformation


class fhn_transformation(transformation):

	def transform(self, X_prev):
		"""
		Integrates the fhn ODEs

		Input:
		fhn_params = a, b, c, I, dt
		a: the shape of the cubic parabola
		b: describes the kinetics of the recovery variable w
		c: describes the kinetics of the recovery variable
		I: input current
		dt: timestep

		Output
		X = [V, w]
			V - membrane voltage
			w - recovery variable that mimics activation of an outward current
		"""
		a, b, c, I, dt = self.params

		def fhn_equation(X, t, a, b, c, I):
			V, w = X
			dVdt = V - V ** 3 / 3 - w + I
			dwdt = a * (b * V - c * w)
			return [dVdt, dwdt]

		t = np.arange(0, 2 * dt, dt)
		X = odeint(fhn_equation, X_prev, t, args=(a, b, c, I))[1, :]

		return X


class tf_fhn_transformation(transformation):

	def transform(self, X_prev):
		"""
		X_prev.shape = [B0, B1, ..., Bn, Dx]
		"""
		a, b, c, I, dt = self.params

		def fhn_equation(X, t):
			V, w = tf.unstack(X, axis=-1)
			dVdt = V - V ** 3 / 3 - w + I
			dwdt = a * (b * V - c * w)
			return tf.stack([dVdt, dwdt], axis=-1)

		t = np.arange(0.0, 2 * dt, dt)
		X = tf.unstack(tf_odeint(fhn_equation, X_prev, t, name="loc"), axis=0)[1]

		return X


# test code
if __name__ == "__main__":
	import matplotlib.pyplot as plt
	fhn_params = (1.0, 0.95, 0.05, 1.0, 0.15)
	Dx = 2
	T = 20
	batch_size = 10

	# for np ver
	fhn = fhn_transformation(fhn_params)

	X = np.zeros((T, Dx))
	X[0] = np.random.uniform(low=0, high=1, size=Dx)
	for t in range(1, T):
		X[t] = fhn.transform(X[t - 1])

	plt.figure()
	plt.plot(X[:, 0], X[:, 1])
	plt.show()

	# for tf ver
	tf_fhn = tf_fhn_transformation(fhn_params)

	Xs = []
	X = tf.constant(np.random.uniform(low=-1, high=1, size=(batch_size, Dx)), dtype=tf.float32)
	for t in range(1, T):
		X = tf_fhn.transform(X)
		Xs.append(X)

	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()
	sess.run(init)
	Xs = tf.stack(Xs, axis=1).eval()

	plt.figure()
	for i in range(batch_size):
		plt.plot(Xs[i, :, 0], Xs[i, :, 1])
	plt.show()

"""
normal and tf version of: multivariate_normal and element-wise Poisson
"""

import scipy as sp
import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

class multivariate_normal:
	""" 
	Define Gaussian density
		P(x_1) = N(x_0, Sigma)
		P(x_t | x_t-1) = N(A * x_t-1, Sigma)

	self.sample(x_t-1) will return a sample x_t based on x_t-1
		to get initial samples at T = 1, use self.sample(None)

	self.prob(x_t-1, x_t) will return the probability f(x_t | x_t-1)
		to get prob nu(x_1) at T = 1, use self.prob(None, x_1)
	"""
	def __init__(self, A, Sigma = None, x_0 = None):
		# A, Sigma, x_0 should be tensors
		self.A = A
		self.Dx = A.shape[0]
		if Sigma is not None:
			self.Sigma = Sigma
		else:
			self.Sigma = np.eye(self.Dx)

		self.SigmaChol = np.linalg.cholesky(Sigma)

		if x_0 is not None:
			self.x_0 = x_0
		else:
			self.x_0 = np.zeros(self.Dx)

	def sample(self, x_prev = None):
		if x_prev is None:
			return self.x_0 + np.dot(self.SigmaChol, np.random.randn(self.Dx))
		else:
			return np.dot(self.A, x_prev) + np.dot(self.SigmaChol, np.random.randn(self.Dx))

	def prob(self, x_prev, x):
		if x_prev is None:
			return sp.stats.multivariate_normal.pdf(x, self.x_0, self.Sigma)
		else:
			return sp.stats.multivariate_normal.pdf(x, np.dot(self.A, x_prev), self.Sigma)

	def log_prob(self, x_prev, x):
		if x_prev is None:
			return sp.stats.multivariate_normal.logpdf(x, self.x_0, self.Sigma)
		else:
			return sp.stats.multivariate_normal.logpdf(x, np.dot(self.A, x_prev), self.Sigma)

class poisson:
	""" 
	Define Poisson density with independent rates y_t ~ Poisson(exp(B * X_t))

	self.sample(x_t) will return a sample y_t based on x_t

	self.prob(x_t, y_t) will return the probability p(y_t | x_t)
	"""	
	def __init__(self, B):
		# B: tensor, shape = (Dy, Dx)
		self.B = B
		self.Dy, self.Dx = B.shape

	def sample(self, x):
		# x: tensor, shape = (n_particles, Dx)
		return np.random.poisson(np.exp(np.dot(self.B,x)))

	def prob(self, x, y):
		assert len(x) == self.Dx
		assert len(y) == self.Dy
		lambdas = np.exp(np.dot(self.B, x))
		element_wise_prob = sp.stats.poisson.pmf(y, lambdas)
		prob = np.prod(element_wise_prob)
		return prob

	def log_prob(self, x, y):
		assert len(x) == self.Dx
		assert len(y) == self.Dy
		lambdas = np.exp(np.dot(self.B, x))
		element_wise_log_prob = sp.stats.poisson.logpmf(y, lambdas)
		log_prob = np.sum(element_wise_log_prob)
		return log_prob

class tf_multivariate_normal:
	""" 
	Define Gaussian density
		P(x_1) = N(x_0, Sigma)
		P(x_t | x_t-1) = N(A * x_t-1, Sigma)

	self.sample(x_t-1) will return a sample x_t based on x_t-1
		to get initial samples at T = 1, use self.sample(None)

	self.prob(x_t-1, x_t) will return the probability f(x_t | x_t-1)
		to get prob nu(x_1) at T = 1, use self.prob(None, x_1)
	"""
	def __init__(self, n_particles, batch_size, A, Sigma = None, x_0 = None, name = 'tf_multivariate_normal', dtype = tf.float32):
		# A: tensor, shape = (Dx, Dx)
		# Sigma: tensor, shape = (Dx, Dx)
		# x_0: tensor, shape = (Dx,)
		with tf.name_scope(name):
			self.n_particles = n_particles
			self.batch_size = batch_size
			self.A = tf.identity(A, name = 'A')
			self.Dx = A.get_shape().as_list()[0]

			if Sigma is not None:
				self.Sigma = tf.identity(Sigma, name = 'Sigma')
			else:
				self.Sigma = tf.eye(self.Dx, name = 'Sigma')			

			if x_0 is not None:
				self.x_0 = tf.identity(x_0, name = 'x_0')
			else:
				self.x_0 = tf.zeros(self.Dx, dtype = dtype, name = 'x_0')

			self.name = name
			self.dtype = dtype

	def get_mvn(self, x_prev, name):
		with tf.name_scope(self.name):
			if x_prev is None:
				mvn = tfd.MultivariateNormalFullCovariance(loc = self.x_0, 
														   covariance_matrix = self.Sigma,
														   name = "mvn")
			else:
				x_prev_r = tf.reshape(x_prev, (self.n_particles*self.batch_size, self.Dx), name = 'x_prev_reshape')
				loc_r = tf.matmul(x_prev_r, self.A, transpose_b = True, name = 'loc_reshape')
				loc = tf.reshape(loc_r, (self.n_particles, self.batch_size, self.Dx), name = 'loc')
				mvn = tfd.MultivariateNormalFullCovariance(loc = loc, 
														   covariance_matrix = self.Sigma,
														   name = "mvn")
			return mvn

	def sample(self, x_prev, name = None):
		# x_prev: tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# sample: tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		if name is None:
			name = self.name
		mvn = self.get_mvn(x_prev, name)
		with tf.name_scope(name):
			if x_prev is None:
				return mvn.sample((self.n_particles, self.batch_size), name = "samples")
			else:
				return mvn.sample(name = "samples")

	def prob(self, x_prev, x, name = None):
		# x_prev: tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# x: tensor,      shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# prob:	tensor,   shape = (n_particles, batch_size), dtype = self.dtype
		if name is None:
			name = self.name
		mvn = self.get_mvn(x_prev, name)
		with tf.name_scope(name):
			if x_prev is None:
				return mvn.prob(x, name = "prob")
			else:
				return mvn.prob(x, name = "prob")


class tf_poisson:
	""" 
	Define Poisson density with independent rates y_t ~ Poisson(exp(B * X_t))

	self.sample(x_t) will return a sample y_t based on x_t

	self.prob(x_t, y_t) will return the probability p(y_t | x_t)
	"""
	
	def __init__(self, n_particles, batch_size, B, name = 'tf_poisson', dtype = tf.float32):
		with tf.name_scope(name):
			self.n_particles = n_particles
			self.batch_size = batch_size
			self.B = tf.identity(B, name = 'B')
			self.Dy, self.Dx = B.get_shape().as_list()
			self.name = name
			self.dtype = dtype

	def sample(self, x, name = None):
		# x: 		tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# sample: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		if name is None:
			name = self.name
		with tf.name_scope(name):
			x_r = tf.reshape(x, (self.n_particles*self.batch_size, self.Dx), name = 'x_reshape')
			log_rate_r = tf.matmul(x_r, self.B, transpose_b = True, name = 'log_rate_reshape')
			log_rate = tf.reshape(log_rate_r, (self.n_particles, self.batch_size, self.Dy), name = 'log_rate')
			poisson = tfd.Poisson(log_rate = log_rate, name = "Poisson")
			return poisson.sample(name = "sample")

	def prob(self, x, y, name = None):
		# x: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# y: 	tensor, shape = (batch_size, Dy,), dtype = self.dtype
		# prob:	tensor, shape = (n_particles, batch_size,), dtype = self.dtype
		if name is None:
			name = self.name
		with tf.name_scope(name):
			y_tile = tf.tile(tf.expand_dims(y, axis = 0), [self.n_particles, 1, 1], name = 'y_tile')
			x_r = tf.reshape(x, (self.n_particles*self.batch_size, self.Dx), name = 'x_reshape')
			log_rate_r = tf.matmul(x_r, self.B, transpose_b = True, name = 'log_rate_reshape')
			log_rate = tf.reshape(log_rate_r, (self.n_particles, self.batch_size, self.Dy), name = 'log_rate')
			poisson = tfd.Poisson(log_rate = log_rate, name = "Poisson")
			return tf.reduce_sum(poisson.prob(y_tile, name = "element_wise_prob"), axis = 2, name = "prob")

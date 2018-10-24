"""
normal and tf version of: mvn and element-wise Poisson
"""

import scipy as sp
import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd
# import tensorflow.contrib.distributions as tfd

class mvn:
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
			return sp.stats.mvn.pdf(x, self.x_0, self.Sigma)
		else:
			return sp.stats.mvn.pdf(x, np.dot(self.A, x_prev), self.Sigma)

	def log_prob(self, x_prev, x):
		if x_prev is None:
			return sp.stats.mvn.logpdf(x, self.x_0, self.Sigma)
		else:
			return sp.stats.mvn.logpdf(x, np.dot(self.A, x_prev), self.Sigma)

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
		# lambdas = np.exp(np.dot(self.B, x))
		lambdas = np.log(np.exp(np.dot(self.B, x)) + 1)
		return np.random.poisson(lambdas)

	def prob(self, x, y):
		# lambdas = np.exp(np.dot(self.B, x))
		lambdas = np.log(np.exp(np.dot(self.B, x)) + 1)
		element_wise_prob = sp.stats.poisson.pmf(y, lambdas)
		prob = np.prod(element_wise_prob)
		return prob

	def log_prob(self, x, y):
		# lambdas = np.exp(np.dot(self.B, x))
		lambdas = np.log(np.exp(np.dot(self.B, x)) + 1)
		element_wise_log_prob = sp.stats.poisson.logpmf(y, lambdas)
		log_prob = np.sum(element_wise_log_prob)
		return log_prob

class tf_mvn:
	""" 
	Define Gaussian density
		P(x_1) = N(x_0, Sigma)
		P(x_t | x_t-1) = N(A * x_t-1, Sigma)

	self.sample(x_t-1) will return a sample x_t based on x_t-1
		to get initial samples at T = 1, use self.sample(None)

	self.prob(x_t-1, x_t) will return the probability f(x_t | x_t-1)
		to get prob nu(x_1) at T = 1, use self.prob(None, x_1)
	"""
	def __init__(self, n_particles, A, Sigma = None, x_0 = None, name = 'tf_mvn', dtype = tf.float32):
		# A: tensor, shape = (Dx, Dx)
		# Sigma: tensor, shape = (Dx, Dx)
		# x_0: tensor, shape = (Dx,)
		with tf.name_scope(name):
			self.n_particles = n_particles
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

	def get_mvn(self, x_prev, name = None):
		with tf.name_scope(name or self.name):
			if x_prev is None:
				return tfd.MultivariateNormalFullCovariance(loc = self.x_0, 
															covariance_matrix = self.Sigma,
															name = "mvn")
			else:
				loc = tf.matmul(x_prev, self.A, transpose_b = True, name = 'loc')
				return tfd.MultivariateNormalFullCovariance(loc = loc, 
															covariance_matrix = self.Sigma,
															name = "mvn")


	def sample(self, x_prev, name = None):
		# x_prev: tensor, shape = (n_particles, Dx), dtype = self.dtype
		# sample: tensor, shape = (n_particles, Dx), dtype = self.dtype
		mvn = self.get_mvn(x_prev, name)
		with tf.name_scope(name or self.name):
			if x_prev is None:
				return mvn.sample(self.n_particles, name = "samples")
			else:
				return mvn.sample(name = "samples")

	def prob(self, x_prev, x, name = None):
		# x_prev: tensor, shape = (n_particles, Dx), dtype = self.dtype
		# x: tensor,      shape = (n_particles, Dx), dtype = self.dtype
		# prob:	tensor, shape = (n_particles,), dtype = self.dtype
		mvn = self.get_mvn(x_prev, name)
		with tf.name_scope(name or self.name):
			return mvn.prob(x, name = "prob")

	def log_prob(self, x_prev, x, name = None):
		# x_prev: tensor, shape = (n_particles, Dx), dtype = self.dtype
		# x: tensor,      shape = (n_particles, Dx), dtype = self.dtype
		# prob:	tensor, shape = (n_particles,), dtype = self.dtype
		mvn = self.get_mvn(x_prev, name)
		with tf.name_scope(name or self.name):
			return mvn.log_prob(x, name = "log_prob")


class tf_poisson:
	""" 
	Define Poisson density with independent rates y_t ~ Poisson(exp(B * X_t))

	self.sample(x_t) will return a sample y_t based on x_t

	self.prob(x_t, y_t) will return the probability p(y_t | x_t)
	"""
	
	def __init__(self, n_particles, B, name = 'tf_poisson', dtype = tf.float32):
		with tf.name_scope(name):
			self.n_particles = n_particles
			self.B = tf.identity(B, name = 'B')
			self.Dy, self.Dx = B.get_shape().as_list()
			self.name = name
			self.dtype = dtype

	def get_poisson(self, x, name = None):
		with tf.name_scope(name or self.name):
			# log_rate = tf.matmul(x, self.B, transpose_b = True, name = 'log_rate')
			# poisson = tfd.Poisson(log_rate = log_rate, name = "Poisson")
			rate = tf.log(tf.exp(tf.matmul(x, self.B, transpose_b = True)) + 1, name = 'rate')
			poisson = tfd.Poisson(rate = rate, name = "Poisson")
			return poisson

	def sample(self, x, name = None):
		# x: 		tensor, shape = (n_particles, Dx), dtype = self.dtype
		# sample: 	tensor, shape = (n_particles, Dx), dtype = self.dtype
		poisson = self.get_poisson(x, name)
		with tf.name_scope(name or self.name):
			return poisson.sample(name = "sample")

	def prob(self, x, y, name = None):
		# x: 	tensor, shape = (n_particles, Dx), dtype = self.dtype
		# y: 	tensor, shape = (Dy,), dtype = self.dtype
		# prob:	tensor, shape = (n_particles,), dtype = self.dtype
		poisson = self.get_poisson(x, name)
		with tf.name_scope(name or self.name):
			return tf.reduce_prod(poisson.prob(y, name = "element_wise_prob"), axis = 1, name = "prob")

	def log_prob(self, x, y, name = None):
		# x: 	tensor, shape = (n_particles, Dx), dtype = self.dtype
		# y: 	tensor, shape = (Dy,), dtype = self.dtype
		# prob:	tensor, shape = (n_particles,), dtype = self.dtype
		poisson = self.get_poisson(x, name)
		with tf.name_scope(name or self.name):
			return tf.reduce_sum(poisson.log_prob(y, name = "element_wise_log_prob"), axis = 1, name = "log_prob")

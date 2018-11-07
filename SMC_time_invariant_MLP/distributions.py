"""
normal and tf version of: multivariate_normal and element-wise Poisson
"""

import scipy as sp
import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

class mvn:
	""" 
	Define multivariate normal density
		P(output | Input) = N(A * Input, Sigma)
		if Input is None, P(output) = N(output_0, Sigma)
	"""
	def __init__(self, A, Sigma = None, output_0 = None):
		self.A = A
		self.Dout, self.Din = A.shape
		if Sigma is not None:
			self.Sigma = Sigma
		else:
			self.Sigma = np.eye(self.Dout)

		self.SigmaChol = np.linalg.cholesky(Sigma)

		if output_0 is not None:
			self.output_0 = output_0
		else:
			self.output_0 = np.zeros(self.Dout)

	def sample(self, Input = None):
		"""
		sample from output ~ N(A * Input, Sigma)
		if Input is None, sample from output ~ N(output_0, Sigma)
		"""
		if Input is None:
			return self.output_0 + np.dot(self.SigmaChol, np.random.randn(self.Dout))
		else:
			return np.dot(self.A, Input) + np.dot(self.SigmaChol, np.random.randn(self.Dout))

	def prob(self, Input, output):
		"""
		return the probability p(output | Input)
		if Input is None, return the probability p(output | output_0)
		"""
		if Input is None:
			return sp.stats.multivariate_normal.pdf(output, self.output_0, self.Sigma)
		else:
			return sp.stats.multivariate_normal.pdf(output, np.dot(self.A, Input), self.Sigma)

	def log_prob(self, Input, output):
		if Input is None:
			return sp.stats.multivariate_normal.logpdf(output, self.output_0, self.Sigma)
		else:
			return sp.stats.multivariate_normal.logpdf(output, np.dot(self.A, Input), self.Sigma)

class poisson:
	""" 
	Define Poisson density with independent rates output ~ Poisson(exp(B * Input))

	self.sample(Input) will sample from output ~ Poisson(B*input)

	self.prob(Input, output) will return the probability p(output | Input)
	"""	
	def __init__(self, B):
		self.B = B
		self.Dout, self.Din = B.shape

	def get_lambdas(self, x):
		# lambdas = np.exp(np.dot(self.B, x))
		lambdas = np.log(np.exp(np.dot(self.B, x)) + 2)
		return lambdas

	def sample(self, x):
		# x: tensor, shape = (n_particles, Dx)
		lambdas = self.get_lambdas(x)
		return np.random.poisson(lambdas)

	def prob(self, x, y):
		lambdas = self.get_lambdas(x)
		element_wise_prob = sp.stats.poisson.pmf(y, lambdas)
		prob = np.prod(element_wise_prob)
		return prob

	def log_prob(self, x, y):
		lambdas = self.get_lambdas(x)
		element_wise_log_prob = sp.stats.poisson.logpmf(y, lambdas)
		log_prob = np.sum(element_wise_log_prob)
		return log_prob

class tf_mvn:
	""" 
	Define multivariate normal density
		P(output | Input) = N(A * Input, Sigma)
		if Input is None, P(output) = N(output_0, Sigma)
	"""
	def __init__(self, n_particles, batch_size, A, Sigma = None, output_0 = None, name = 'tf_mvn', dtype = tf.float32):
		# A: tensor, shape = (Dout, Din)
		# Sigma: tensor, shape = (Dout, Dout)
		# output_0: tensor, shape = (Dout,)
		with tf.name_scope(name):
			self.n_particles = n_particles
			self.batch_size = batch_size
			self.A = tf.identity(A, name = 'A')
			self.Dout, self.Din = A.get_shape().as_list()

			if Sigma is not None:
				self.Sigma = tf.identity(Sigma, name = 'Sigma')
			else:
				self.Sigma = tf.eye(self.Dout, name = 'Sigma')			

			if output_0 is not None:
				self.output_0 = tf.identity(output_0, name = 'output_0')
			else:
				self.output_0 = tf.zeros(self.Dout, dtype = dtype, name = 'output_0')

			self.name = name
			self.dtype = dtype

	def get_mvn(self, Input, name):
		with tf.name_scope(self.name):
			if Input is None:
				mvn = tfd.MultivariateNormalFullCovariance(loc = self.output_0, 
														   covariance_matrix = self.Sigma,
														   validate_args=True,
														   allow_nan_stats=False, 
														   name = "mvn")
			else:
				Input_r = tf.reshape(Input, (self.n_particles*self.batch_size, self.Din), name = 'Input_r')
				loc_r = tf.matmul(Input_r, self.A, transpose_b = True, name = 'loc_reshape')
				loc = tf.reshape(loc_r, (self.n_particles, self.batch_size, self.Dout), name = 'loc')
				mvn = tfd.MultivariateNormalFullCovariance(loc = loc, 
														   covariance_matrix = self.Sigma,
														   validate_args=True,
														   allow_nan_stats=False, 
														   name = "mvn")
			return mvn

	def sample(self, Input, name = None):
		# Input:  tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# sample: tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		mvn = self.get_mvn(Input, name)
		with tf.name_scope(name or self.name):
			if Input is None:
				return mvn.sample(self.n_particles, name = "samples")
			else:
				return mvn.sample(name = "samples")

	def prob(self, Input, output, name = None):
		# Input: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# output: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# prob:		tensor, shape = (n_particles, batch_size), dtype = self.dtype
		mvn = self.get_mvn(Input, name)
		with tf.name_scope(name or self.name):
			if Input is None:
				return mvn.prob(output, name = "prob")
			else:
				return mvn.prob(output, name = "prob")

	def log_prob(self, Input, output, name = None):
		# Input: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# output: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# prob:		tensor, shape = (n_particles, batch_size), dtype = self.dtype
		mvn = self.get_mvn(Input, name)
		with tf.name_scope(name or self.name):
			if Input is None:
				return mvn.log_prob(output, name = "log_prob")
			else:
				return mvn.log_prob(output, name = "log_prob")

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
			self.Dout, self.Din = B.get_shape().as_list()
			self.name = name
			self.dtype = dtype

	def get_poisson(self, Input, name = None):
		with tf.name_scope(name or self.name):
			Input_r = tf.reshape(Input, (self.n_particles*self.batch_size, self.Din), name = 'Input_reshape')
			log_rate_r = tf.matmul(Input_r, self.B, transpose_b = True, name = 'log_rate_reshape')
			# log_rate = tf.reshape(log_rate_r, (self.n_particles, self.batch_size, self.Dout), name = 'log_rate')
			# poisson = tfd.Poisson(log_rate = log_rate, validate_args=False, name = "Poisson")
			rate_r = tf.log(2 + tf.exp(log_rate_r))
			rate = tf.reshape(rate_r, (self.n_particles, self.batch_size, self.Dout), name = 'rate')
			poisson = tfd.Poisson(rate = rate, 
								  validate_args=True, 
								  allow_nan_stats=False, 
								  name = "Poisson")
			return poisson

	def sample(self, Input, name = None):
		# Input: 	tensor, shape = (n_particles, batch_size, Din), dtype = self.dtype
		# sample: 	tensor, shape = (n_particles, batch_size, Dout), dtype = self.dtype
		poisson = self.get_poisson(Input, name)
		with tf.name_scope(name or self.name):
			return poisson.sample(name = "sample")

	def prob(self, Input, output, name = None):
		# Input: 	tensor, shape = (n_particles, batch_size, Din), dtype = self.dtype
		# output: 	tensor, shape = (batch_size, Dy), dtype = self.dtype
		# prob:		tensor, shape = (n_particles, batch_siz), dtype = self.dtype
		poisson = self.get_poisson(Input, name)
		with tf.name_scope(name or self.name):
			return tf.reduce_prod(poisson.prob(output, name = "element_wise_prob"), axis = 2, name = "prob")

	def log_prob(self, Input, output, name = None):
		# Input: 	tensor, shape = (n_particles, batch_size, Din), dtype = self.dtype
		# output: 	tensor, shape = (batch_size, Dy), dtype = self.dtype
		# prob:		tensor, shape = (n_particles, batch_siz), dtype = self.dtype
		poisson = self.get_poisson(Input, name)
		with tf.name_scope(name or self.name):
			return tf.reduce_sum(poisson.log_prob(output, name = "element_wise_log_prob"), axis = 2, name = "log_prob")

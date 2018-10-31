import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.integrate import odeint as tf_odeint


class tf_andrieu_transition:
	""" 
	Define multivariate normal density based on andrieu_transition model
	"""
	def __init__(self, n_particles, batch_size, theta1, Sigma = None, output_0 = None, name = 'tf_fhn'):
		with tf.name_scope(name):
			self.n_particles = n_particles
			self.batch_size = batch_size
			self.theta1 = theta1
			self.Sigma = Sigma or 10*tf.eye(1, name = 'Sigma')
			self.output_0 = output_0 or tf.constant(np.random.randn(batch_size, 1) * np.sqrt(5), dtype = tf.float32)
			self.Dout = self.Din = 1
			self.name = name

	def get_mvn(self, Input, t, name):
		with tf.name_scope(self.name):
			if Input is None:
				mvn = tfd.MultivariateNormalFullCovariance(loc = self.output_0, 
														   covariance_matrix = self.Sigma,
														   validate_args=True,
														   allow_nan_stats=False, 
														   name = "fhn_mvn")
			else:
				loc = self.theta1 * Input + 25.0*Input/(1.0+Input**2) + 8.0*np.sqrt(10) * np.cos(1.2*t)
				mvn = tfd.MultivariateNormalFullCovariance(loc = loc, 
														   covariance_matrix = self.Sigma,
														   validate_args=True,
														   allow_nan_stats=False, 
														   name = "fhn_mvn")
			return mvn

	def sample(self, Input, t = 0, name = None):
		# Input:  tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# sample: tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		mvn = self.get_mvn(Input, t, name)
		with tf.name_scope(name or self.name):
			if Input is None:
				return mvn.sample((self.n_particles), name = "samples")
			else:
				return mvn.sample(name = "samples")

	def prob(self, Input, output, t = 0, name = None):
		# Input: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# output: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# prob:		tensor, shape = (n_particles, batch_size), dtype = self.dtype
		mvn = self.get_mvn(Input, t, name)
		with tf.name_scope(name or self.name):
			if Input is None:
				return mvn.prob(output, name = "prob")
			else:
				return mvn.prob(output, name = "prob")

	def log_prob(self, Input, output, t = 0, name = None):
		# Input: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# output: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
		# prob:		tensor, shape = (n_particles, batch_size), dtype = self.dtype
		mvn = self.get_mvn(Input, t, name)
		with tf.name_scope(name or self.name):
			if Input is None:
				return mvn.log_prob(output, name = "log_prob")
			else:
				return mvn.log_prob(output, name = "log_prob")


class tf_andrieu_emission:
	""" 
	Define multivariate normal density based on andrieu_emission model
	"""
	def __init__(self, n_particles, batch_size, theta2, Sigma = None, name = 'tf_fhn'):
		with tf.name_scope(name):
			self.n_particles = n_particles
			self.batch_size = batch_size
			self.theta2 = theta2
			self.Sigma = Sigma or tf.constant(100, dtype = tf.float32, shape = [1, 1], name = 'Sigma')
			self.Dout = self.Din = 1
			self.name = name

	def get_mvn(self, Input, name):
		with tf.name_scope(self.name):
			loc = self.theta2 * Input**2
			mvn = tfd.MultivariateNormalFullCovariance(loc = loc, 
													   covariance_matrix = self.Sigma,
													   validate_args=True,
													   allow_nan_stats=False, 
													   name = "fhn_mvn")
			return mvn

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
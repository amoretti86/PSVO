import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from distribution.base import distribution

# np ver, just used in sampler, so no need to implement log_prob
class mvn(distribution):
	"""
	multivariate normal distribution
	"""
	def __init__(self, transformation, sigma):
		self.sigmaChol = np.linalg.cholesky(sigma)
		self.transformation = transformation

	def sample(self, Input):
		mu = self.transformation.transform(Input)
		return mu + np.dot(self.sigmaChol, np.random.randn(len(mu)))

# tf ver, used in calculate log_ZSMC
class tf_mvn(distribution):
	"""
	multivariate normal distribution
	"""
	def __init__(self, transformation,
				 output_0=None,
				 sigma=None, sigma_init = 5, sigma_min = 1,
				 name = 'tf_mvn'):
		self.transformation = transformation
		self.output_0 = output_0
		self.sigma = sigma
		self.sigma_init = sigma_init
		self.sigma_min = sigma_min
		self.name = name

	def get_mvn(self, Input = None):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			if Input is None:
				if self.output_0 is None:
					raise ValueError("output_0 is not initialized for {}!".format(self.name))
				mu = self.output_0
			else:
				mu = self.transformation.transform(Input)

			if self.sigma != None:
				sigma_tmp = tf.identity(self.sigma, name = "sigma_tmp")
			else:
				Dout = mu.shape.as_list()[-1]
				sigma_tmp = tf.get_variable("sigma",
											shape = [Dout],
											dtype = tf.float32,
											initializer = tf.constant_initializer(self.sigma_init),
											trainable = True)
				sigma_tmp = tf.maximum(tf.nn.softplus(sigma_tmp), self.sigma_min)
				sigma_tmp = tf.matrix_diag(sigma_tmp, name = "sigma_tmp")

			mvn = tfd.MultivariateNormalFullCovariance(mu, sigma_tmp,
													   validate_args=True,
													   allow_nan_stats=False)
			return mvn

	def sample_and_log_prob(self, Input, sample_shape=(), name=None):
		mvn = self.get_mvn(Input)
		with tf.variable_scope(name or self.name):
			sample = mvn.sample(sample_shape)
			log_prob = mvn.log_prob(sample)
			return sample, log_prob

	def log_prob(self, Input, output, name = None):
		mvn = self.get_mvn(Input)
		with tf.variable_scope(name or self.name):
			return mvn.log_prob(output)

	def mean(self, Input, name = None):
		mvn = self.get_mvn(Input)
		with tf.variable_scope(name or self.name):
			return mvn.mean()
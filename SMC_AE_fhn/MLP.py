import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer

class MLP_mvn:
	def __init__(self, Dx, Dy, sigma_cons = 5, name = 'MLP_mvn'):
		self.Dx = Dx
		self.Dy = Dy
		self.Dh1 = 20

		self.sigma_cons = sigma_cons

		self.name = name

	def get_mvn(self, x):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			sigma_cons = tf.get_variable("sigma_cons",
										 shape = [1],
										 dtype = tf.float32,
										 initializer = tf.constant_initializer(self.sigma_cons),
										 trainable = True)
			Dh1 = fully_connected(x, self.Dh1, 
								  weights_initializer=xavier_initializer(uniform=False), 
								  biases_initializer=tf.constant_initializer(0),
								  activation_fn = tf.nn.relu,
								  reuse = tf.AUTO_REUSE, scope = "Dh1")
			mu = fully_connected(Dh1, self.Dy,
								 weights_initializer=xavier_initializer(uniform=False), 
								 biases_initializer=tf.constant_initializer(0),
								 activation_fn = None,
								 reuse = tf.AUTO_REUSE, scope = "mu")
			sigma = fully_connected(Dh1, self.Dy,
									weights_initializer=xavier_initializer(uniform=False), 
									biases_initializer=tf.constant_initializer(0.6),
									activation_fn = tf.nn.softplus,
									reuse = tf.AUTO_REUSE, scope = "sigma") + sigma_cons
			mvn = tfd.MultivariateNormalFullCovariance(loc = mu, 
													   covariance_matrix = tf.matrix_diag(sigma), 
													   validate_args=True, 
													   name = "mvn")
			return mvn

	def sample(self, x, name = None):
		mvn = self.get_mvn(x)
		with tf.variable_scope(name or self.name):
			return mvn.sample(name = 'sample')

	def prob(self, x, y, name = None):
		mvn = self.get_mvn(x)
		with tf.variable_scope(name or self.name):
			return mvn.prob(y, name = 'prob')

	def log_prob(self, x, y, name = None):
		mvn = self.get_mvn(x)
		with tf.variable_scope(name or self.name):
			return mvn.log_prob(y, name = 'log_prob')



class MLP_poisson:
	def __init__(self, Dx, Dy, lambda_cons = 1e-2, name = 'MLP_poisson'):
		self.Dx = Dx
		self.Dy = Dy
		self.Dh1 = 50

		self.lambda_cons = lambda_cons

		self.name = name

	def get_poisson(self, x):
		with tf.variable_scope(self.name):
			Dh1 = fully_connected(x, self.Dh1, 
								  weights_initializer=xavier_initializer(uniform=False), 
								  biases_initializer=tf.constant_initializer(0),
								  activation_fn = tf.nn.relu,
								  reuse = tf.AUTO_REUSE, scope = "Dh1")
			lambdas = fully_connected(Dh1, self.Dy,
									  weights_initializer=xavier_initializer(uniform=False), 
									  biases_initializer=tf.constant_initializer(0.6),
									  activation_fn = tf.nn.softplus,
									  reuse = tf.AUTO_REUSE, scope = "lambdas") + self.lambda_cons
			poisson = tfd.Poisson(rate = lambdas, validate_args=True, name = "Poisson")
			return poisson

	def sample(self, x, name = None):
		poisson = self.get_poisson(x)
		with tf.variable_scope(name or self.name):
			return poisson.sample(name = 'sample')

	def prob(self, x, y, name = None):
		poisson = self.get_poisson(x)
		with tf.variable_scope(name or self.name):
			return tf.reduce_prod(poisson.prob(y, name = 'elementwise_prob'), 
								  axis = -1, name = 'prob')

	def log_prob(self, x, y, name = None):
		poisson = self.get_poisson(x)
		with tf.variable_scope(name or self.name):
			return tf.reduce_sum(poisson.log_prob(y, name = 'elementwise_log_prob'), 
								 axis = -1, name = 'log_prob')
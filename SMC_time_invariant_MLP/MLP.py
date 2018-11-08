import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer

class MLP_mvn:
	def __init__(self, Dx, Dy, 
				 n_particles, batch_size,
				 sigma_init = 5,
				 name = 'MLP_mvn'):
		self.Dx = Dx
		self.Dy = Dy
		self.Dh1 = 50

		self.n_particles = n_particles
		self.batch_size = batch_size

		self.sigma_init = sigma_init
		self.sigma_min = 1

		self.name = name

	def get_mvn(self, x):
		with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
			sigma_cons = tf.get_variable("sigma_cons",
										 shape = [self.Dy],
										 dtype = tf.float32,
										 initializer = tf.constant_initializer(self.sigma_init),
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
			# sigma = fully_connected(Dh1, self.Dy,
			# 						weights_initializer=xavier_initializer(uniform=False), 
			# 						biases_initializer=tf.constant_initializer(0.6),
			# 						activation_fn = tf.nn.softplus,
			# 						reuse = tf.AUTO_REUSE, scope = "sigma")
			sigma = tf.maximum(tf.nn.softplus(sigma_cons), self.sigma_min)
			mvn = tfd.MultivariateNormalFullCovariance(loc = mu, 
													   covariance_matrix = tf.matrix_diag(sigma), 
													   validate_args=True, 
													   allow_nan_stats=False, 
													   name = "mvn")
			return mvn

	def sample_and_log_prob(self, x, name = None):
		mvn = self.get_mvn(x)
		with tf.variable_scope(name or self.name):
			if len(x.get_shape().as_list()) == 2:
				sample = mvn.sample(self.n_particles, name = 'sample')
			else:
				sample = mvn.sample(name = 'sample')
			log_prob = mvn.log_prob(sample, name = 'log_prob')
			return sample, log_prob

	def log_prob(self, x, y, name = None):
		mvn = self.get_mvn(x)
		with tf.variable_scope(name or self.name):
			return mvn.log_prob(y, name = 'log_prob')



class MLP_poisson:
	def __init__(self, Dx, Dy,
				 n_particles, batch_size, 
				 lambda_cons = 1e-6, name = 'MLP_poisson'):
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
			poisson = tfd.Poisson(rate = lambdas, 
								  validate_args=True, 
								  allow_nan_stats=False, 
								  name = "Poisson")
			return poisson

	def sample(self, x, name = None):
		poisson = self.get_poisson(x)
		with tf.variable_scope(name or self.name):
			return poisson.sample(name = 'sample')

	def log_prob(self, x, y, name = None):
		poisson = self.get_poisson(x)
		with tf.variable_scope(name or self.name):
			return tf.reduce_sum(poisson.log_prob(y, name = 'elementwise_log_prob'), 
								 axis = -1, name = 'log_prob')
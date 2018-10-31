import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer

class Encoder_full_obs:
	def __init__(self, x_dim, y_dim, n_particles, batch_size, time, scope = 'Encoder_full_obs'):
		self.scope = scope
		self.x_dim = x_dim
		self.y_dim = y_dim
		self.n_particles = n_particles
		self.batch_size = batch_size
		self.time = time

		self.Enc_layer_1_dim = (self.time - 1) * 5
		self.Enc_layer_2_dim = (self.time - 1) * 5
		self.Ev_layer_1_dim = 200
		self.Ev_layer_2_dim = 200

		self.alpha = 0.1
		self.sigma_init = 25
		self.sigma_cons_stablizer = 1e-6

	# ================================ keep an eye on len of YInput ================================ #
	# =================================== len = time or time - 1 =================================== #
	def encoding(self, YInput_NbxTxDy):
		with tf.variable_scope(self.scope + '/encoding'):
			YInput_reshape = tf.reshape(YInput_NbxTxDy, [self.batch_size, self.time*self.y_dim], name = 'YInput_reshape')
			Enc_layer_1 = fully_connected(YInput_reshape, self.Enc_layer_1_dim,
										  weights_initializer=tf.orthogonal_initializer(),
										  biases_initializer=tf.zeros_initializer(),
										  activation_fn=tf.nn.softmax,
										  reuse = tf.AUTO_REUSE, scope = "Enc_layer_1")
			Enc_layer_2 = fully_connected(Enc_layer_1, self.Enc_layer_2_dim,
										  weights_initializer=tf.orthogonal_initializer(),
										  biases_initializer=tf.zeros_initializer(),
										  activation_fn=tf.nn.softmax,
										  reuse = tf.AUTO_REUSE, scope = "Enc_layer_2")
			X_hat_flat = fully_connected(Enc_layer_2, (self.time - 1) * self.x_dim,
											weights_initializer=tf.orthogonal_initializer(),
											biases_initializer=tf.zeros_initializer(),
											activation_fn=None,
											reuse = tf.AUTO_REUSE, scope = "X_hat_flat")
			X_hat_NbxTxDz = tf.reshape(X_hat_flat, [self.batch_size, self.time - 1, self.x_dim], name = "X_hat_NbxTxDz")
			return X_hat_NbxTxDz

	def evolving(self, X_hat_NbxTxDz):
		with tf.variable_scope(self.scope + '/evolving'):
			Ev_layer_1 = fully_connected(X_hat_NbxTxDz, self.Ev_layer_1_dim,
										 weights_initializer=tf.orthogonal_initializer(),
										 biases_initializer=tf.zeros_initializer(),
										 activation_fn=tf.nn.softplus,
										 reuse = tf.AUTO_REUSE, scope = "Ev_layer_1")
			Ev_layer_2 = fully_connected(Ev_layer_1, self.Ev_layer_2_dim,
										 weights_initializer=tf.orthogonal_initializer(),
										 biases_initializer=tf.zeros_initializer(),
										 activation_fn=tf.nn.softplus,
										 reuse = tf.AUTO_REUSE, scope = "Ev_layer_2")
			B_flat  = fully_connected(Ev_layer_2, self.x_dim**2,
									  weights_initializer=tf.orthogonal_initializer(),
									  biases_initializer=tf.zeros_initializer(),
									  activation_fn=None,
									  reuse = tf.AUTO_REUSE, scope = "B_flat")
			# check dim! time or time - 1
			B_NbxTxDzxDz = tf.reshape(B_flat, [self.batch_size, self.time - 1, self.x_dim, self.x_dim])
			# B must be symmetric
			B_NbxTxDzxDz = tf.einsum('btij, btkj->btik', B_NbxTxDzxDz, B_NbxTxDzxDz)
			return B_NbxTxDzxDz

	def get_A(self, YInput_NbxTxDy):
		X_hat_NbxTxDz = self.encoding(YInput_NbxTxDy)
		B_NbxTxDzxDz = self.evolving(X_hat_NbxTxDz)
		with tf.variable_scope(self.scope + '/get_A'):
			self.A_NbxTxDzxDz = tf.add(tf.eye(self.x_dim), self.alpha * B_NbxTxDzxDz, name = 'A_NxTxDzxDz')
			self.A_NbxDzxDz_list = tf.unstack(self.A_NbxTxDzxDz, axis = 1, name = 'A_NxDzxDz_list')
			return self.A_NbxTxDzxDz

	def get_mvn(self, x_prev_NpxNbxDz, t):
		with tf.variable_scope(self.scope + '/get_mvn', reuse=tf.AUTO_REUSE):
			# check idx! t or t - 1
			A_NbxDzxDz = self.A_NbxDzxDz_list[t - 1]
			mu = tf.einsum('bjk, pbk->pbj', A_NbxDzxDz, x_prev_NpxNbxDz, name = 'mu')
			sigma_cons = tf.get_variable("sigma_cons",
										 shape = [1],
										 dtype = tf.float32,
										 initializer = tf.constant_initializer(self.sigma_init),
										 trainable = True)
			sigma = tf.eye(self.x_dim) * (tf.nn.softplus(sigma_cons) + self.sigma_cons_stablizer)
			mvn = tfd.MultivariateNormalFullCovariance(loc = mu, 
													   covariance_matrix = sigma,
													   validate_args=True, 
													   name = "mvn")
			return mvn

	def prob(self, x_prev_NpxNbxDz, x_NpxNbxDz, t, name = None):
		mvn = self.get_mvn(x_prev_NpxNbxDz, t)
		with tf.variable_scope(name or self.scope):
			return mvn.prob(x_NpxNbxDz, name = 'prob')

	def log_prob(self, x_prev_NpxNbxDz, x_NpxNbxDz, t, name = None):
		mvn = self.get_mvn(x_prev_NpxNbxDz, t)
		with tf.variable_scope(name or self.scope):
			return mvn.log_prob(x_NpxNbxDz, name = 'log_prob')

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer

class VartiationalRNN():

	def __init__(self, x_dim, y_dim, h_dim, n_particles, batch_size, sigma_cons = 0.6, variable_scope = "VRNN"):
		self.Dh = h_dim
		self.Dx = x_dim
		self.Dy = y_dim
		# x feature extraction
		self.Dx_1 = x_dim
		# y feature extraction
		self.Dy_1 = y_dim

		self.n_particles = n_particles
		self.batch_size = batch_size
		self.sigma_cons = sigma_cons
		self.variable_scope = variable_scope

		self.lstm = tf.nn.rnn_cell.LSTMCell(h_dim, state_is_tuple=True)
		self.reset()

	def reset(self):
		# lstm return to zero_state
		with tf.variable_scope(self.variable_scope + '/reset'):
			self.state = self.lstm.zero_state((self.n_particles * self.batch_size), dtype = tf.float32)
			_,  self.h = self.state

			# self.h_stack.shape = (n_particles, batch_size, self.Dh)
			self.h_stack = tf.reshape(self.h, (self.n_particles, self.batch_size, self.Dh), name = 'h_stack')

	def get_x_ft(self, x):
		# x feature extraction
		with tf.variable_scope(self.variable_scope + '/get_x_ft'):
			x_1 = fully_connected(x, self.Dx_1, reuse = tf.AUTO_REUSE, scope = 'x_to_x_1')
			x_ft = tf.identity(x_1, name = 'x_ft')
		return x_ft

	def get_y_ft(self, y):
		# y feature extraction
		with tf.variable_scope(self.variable_scope + '/get_y_ft'):
			y_1 = fully_connected(y, self.Dy_1, reuse = tf.AUTO_REUSE, scope = "y_to_y_1")
			y_ft = tf.identity(y_1, name = 'y_ft')
		return y_ft

	def get_q(self, h_stack, y_t_ft):
		"""
		calculate x_t ~ q(*|h_t-1, y_t_ft)
		h_stack.shape = (n_particles, batch_size, Dh)
		y_t_ft.shape  = (batch_size,  Dy_1)
		"""
		with tf.variable_scope(self.variable_scope + '/get_q'):
			y_t_ft_expanded = tf.expand_dims(y_t_ft, axis = 0, name = 'y_t_ft_expanded')
			# y_t_ft_expanded.shape = (1, batch_size, Dy_1)
			y_t_ft_tiled = tf.tile(y_t_ft_expanded, (self.n_particles, 1, 1), name = 'y_t_ft_tiled')
		 	# y_t_ft_tiled.shape 	= (n_paticles, batch_size, Dy_1)
			h_y_concat = tf.concat((h_stack, y_t_ft_tiled), axis = 2, name = 'h_y_concat')
			# h_y_concat.shape 		= (n_paticles, batch_size, Dh + Dy_1)
			mu    = fully_connected(h_y_concat, self.Dx, 
									weights_initializer=xavier_initializer(uniform=False), 
									activation_fn = None, 
									reuse = tf.AUTO_REUSE, scope = "mu")
			# mu.shape 				= (n_paticles, batch_size, Dx)
			sigma = fully_connected(h_y_concat, self.Dx,
									weights_initializer=xavier_initializer(uniform=False), 
									biases_initializer=tf.constant_initializer(0.6),
									activation_fn = tf.nn.softplus, 
									reuse = tf.AUTO_REUSE, scope = "sigma") + self.sigma_cons
			# sigma.shape 			= (n_paticles, batch_size, Dx)
			q = tfd.MultivariateNormalFullCovariance(loc = mu, covariance_matrix = tf.matrix_diag(sigma), 
													 name = "q")
			return q 

	def get_f(self, h_stack):
		"""
		calculate x_t ~ f(*|h_t-1)
		h_stack.shape = (n_particles, batch_size, Dh)
		"""
		with tf.variable_scope(self.variable_scope + '/get_f'):
			mu    = fully_connected(h_stack, self.Dx, 
									weights_initializer=xavier_initializer(uniform=False), 
									activation_fn = None, 
									reuse = tf.AUTO_REUSE, scope = "mu")
			# mu.shape 				= (n_paticles, batch_size, Dx)
			sigma = fully_connected(h_stack, self.Dx,
									weights_initializer=xavier_initializer(uniform=False), 
									biases_initializer=tf.constant_initializer(0.6),
									activation_fn = tf.nn.softplus, 
									reuse = tf.AUTO_REUSE, scope = "sigma") + self.sigma_cons
			# sigma.shape 			= (n_paticles, batch_size, Dx)
			f = tfd.MultivariateNormalFullCovariance(loc = mu, covariance_matrix = tf.matrix_diag(sigma), 
													 name = "f")
			return f 

	def get_g(self, h_stack, x_t):
		"""
		calculate y_t ~ g(*|h_t-1, x_t)
		h_stack.shape = (n_particles, batch_size, Dh)
		x_t.shape 	  = (n_particles, batch_size, Dx)
		"""
		x_t_ft = self.get_x_ft(x_t)
		# x_t_ft.shape = (n_particles, batch_size, Dx_1)
		with tf.variable_scope(self.variable_scope + '/get_g'):
			h_x_concat = tf.concat((h_stack, x_t_ft), axis = 2, name = 'h_x_concat')
			mu    = fully_connected(h_x_concat, self.Dy, 
									weights_initializer=xavier_initializer(uniform=False), 
									activation_fn = None, 
									reuse = tf.AUTO_REUSE, scope = "mu")
			# mu.shape 				= (n_paticles, batch_size, Dx)
			sigma = fully_connected(h_x_concat, self.Dy,
									weights_initializer=xavier_initializer(uniform=False),
									biases_initializer=tf.constant_initializer(0.6),
									activation_fn = tf.nn.softplus, 
									reuse = tf.AUTO_REUSE, scope = "sigma") + self.sigma_cons
			# sigma.shape 			= (n_paticles, batch_size, Dx)
			g = tfd.MultivariateNormalFullCovariance(loc = mu, covariance_matrix = tf.matrix_diag(sigma), 
													 name = "g")
			return g 

	def step(self, y_t):
		"""
		at each timestep, LSTM take y as input, y_t.shape = (self.batch_size, Dy)
		calculate y_t feature extraction y_t_ft
		calculate q, q(x_t | h_t-1, y_t_ft)
		calculate f, f(x_t | h_t-1)
		"""
		y_t_ft = self.get_y_ft(y_t)
		self.q = self.get_q(self.h_stack, y_t_ft)
		self.f = self.get_f(self.h_stack)

	def get_q_sample(self):
		with tf.variable_scope(self.variable_scope + '/get_q_sample'):
			return self.q.sample(name = 'q_sample')	
			# shape = (n_particles, batch_size, Dx)

	def get_q_prob(self, x):
		"""
		calculate q(x_t|h_t-1, y_t)
		x.shape = (n_particles, batch_size, Dx)
		"""
		with tf.variable_scope(self.variable_scope + '/get_q_prob'):
			# return tf.maximum(self.q.prob(x, name = 'q_true_prob'), 1e-9, name = 'q_clipped_prob')
			return self.q.prob(x, name = 'q_prob')
			# shape = (n_particles, batch_size)

	def get_f_prob(self, x):
		"""
		calculate f(x_t|h_t-1)
		x.shape = (n_particles, batch_size, Dx)
		"""
		with tf.variable_scope(self.variable_scope + '/get_f_prob'):
			# return tf.maximum(self.f.prob(x, name = 'f_true_prob'), 1e-9, name = 'f_clipped_prob')
			return self.f.prob(x, name = 'f_prob')
			# shape = (n_particles, batch_size)

	def get_g_prob(self, x_t, y_t):
		"""
		calculate g(y_t|h_t-1, x_t)
		x_t.shape = (n_particles, batch_size, Dx)
		y_t.shape = (batch_size, Dy)
		"""
		self.g = self.get_g(self.h_stack, x_t)
		with tf.variable_scope(self.variable_scope + '/get_g_prob'):
			# return tf.maximum(self.g.prob(y_t, name = 'g_true_prob'), 1e-9, name = 'g_clipped_prob')
			return self.g.prob(y_t, name = 'g_prob')
			# shape = (n_particles, batch_size)

	def get_q_log_prob(self, x):
		"""
		calculate q(x_t|h_t-1, y_t)
		x.shape = (n_particles, batch_size, Dx)
		"""
		with tf.variable_scope(self.variable_scope + '/get_q_log_prob'):
			# return tf.maximum(self.q.log_prob(x, name = 'q_true_log_prob'), -30.0, name = 'q_clipped_log_prob')
			return self.q.log_prob(x, name = 'q_log_prob')
			# shape = (n_particles, batch_size)

	def get_f_log_prob(self, x):
		"""
		calculate f(x_t|h_t-1)
		x.shape = (n_particles, batch_size, Dx)
		"""
		with tf.variable_scope(self.variable_scope + '/get_f_log_prob'):
			# return tf.maximum(self.f.log_prob(x, name = 'f_true_log_prob'), -30.0, name = 'f_clipped_log_prob')
			return self.f.prob(x, name = 'f_log_prob')
			# shape = (n_particles, batch_size)

	def get_g_log_prob(self, x_t, y_t):
		"""
		calculate g(y_t|h_t-1, x_t)
		x_t.shape = (n_particles, batch_size, Dx)
		y_t.shape = (batch_size, Dy)
		"""
		self.g = self.get_g(self.h_stack, x_t)
		with tf.variable_scope(self.variable_scope + '/get_g_log_prob'):
			return tf.maximum(self.g.log_prob(y_t, name = 'g_true_log_prob'), -30.0, name = 'g_clipped_log_prob')
			return self.g.log_prob(y_t, name = 'g_log_prob')
			# shape = (n_particles, batch_size)

	def update_lstm(self, x_t, y_t):
		"""
		update h_t-1 to h_t
		x_t.shape = (n_particles, batch_size, Dx)
		y_t.shape = (batch_size, Dy)
		"""
		x_t_ft = self.get_x_ft(x_t)
		# X_t_ft.shape = (n_particles, batch_size, Dx_1)
		y_t_ft = self.get_y_ft(y_t)
		# y_t_ft.shape = (batch_size, Dy_1)
		with tf.variable_scope(self.variable_scope + '/update_lstm'):
			y_t_ft_expanded = tf.expand_dims(y_t_ft, axis = 0, name = 'y_t_ft_expanded')
			# y_t_ft_expanded.shape = (1, batch_size, Dy_1)
			y_t_ft_tiled = tf.tile(y_t_ft_expanded, (self.n_particles, 1, 1), name = 'y_t_ft_tiled')
		 	# y_t_ft_tiled.shape 	= (n_particles, batch_size, Dy_1)
			xy_t_ft = tf.concat((x_t_ft, y_t_ft_tiled), axis = 2, name = 'xy_t_ft')
		 	# xy_t_ft.shape 		= (n_particles, batch_size, Dx_1 + Dy_1)
			lstm_input = tf.reshape(xy_t_ft, (self.n_particles * self.batch_size, self.Dx_1 + self.Dy_1))

			_, self.state = self.lstm(lstm_input, self.state)
			_, self.h = self.state

			self.h_stack = tf.reshape(self.h, (self.n_particles, self.batch_size, self.Dh), name = 'h_stack')
			# self.h_stack.shape = (n_particles, batch_size, self.Dh)
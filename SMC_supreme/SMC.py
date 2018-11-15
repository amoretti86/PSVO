import tensorflow as tf
from tensorflow_probability import distributions as tfd

class SMC:
	def __init__(self, q, f, g,
				 n_particles, batch_size,
				 encoder_cell = None,
				 use_stop_gradient = False,
				 name = "log_ZSMC"):
		self.q = q
		self.f = f
		self.g = g
		self.n_particles = n_particles
		self.batch_size = batch_size

		self.encoder_cell = encoder_cell

		self.use_stop_gradient = use_stop_gradient
		self.name = name

	def get_log_ZSMC(self, obs, x_0):
		"""
		Input:
			obs.shape = (batch_size, time, Dy)
		Output:
			log_ZSMC: shape = scalar
			log: stuff to debug
		"""
		with tf.variable_scope(self.name):
			batch_size, time, Dy = obs.get_shape().as_list()
			batch_size, Dx = x_0.get_shape().as_list()

			Xs = []
			log_Ws = []
			Ws = []
			fs = []
			gs = []
			qs = []

			if self.encoder_cell is not None:
				self.encoder_cell.encode(obs[:, 0:-1], x_0)

			log_ZSMC = 0
			X_prev = None

			for t in range(0, time):
				# when t = 1, sample with x_0
				# otherwise, sample with X_prev
				if X_prev is None:
					sample_size = (self.n_particles)
				else:
					sample_size = ()

				X, q_t_log_prob = self.q.sample_and_log_prob(X_prev, sample_shape=sample_size, name="q_{}_log_prob".format(t))
				f_t_log_prob = self.f.log_prob(X_prev, X, name="f_{}_log_prob".format(t))
				g_t_log_prob = self.g.log_prob(X, obs[:,t], name="g_{}_log_prob".format(t))
				
				log_W = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name = "log_W_{}".format(t))
				W = tf.exp(log_W, name = "W_{}".format(t))
				log_ZSMC += tf.log(tf.reduce_mean(W, axis = 0, name = "W_{}_mean".format(t)), name = "log_ZSMC_{}".format(t))

				# print(X)
				# print(q_t_log_prob)
				# print(f_t_log_prob)
				# print(g_t_log_prob)
				# print(log_W)

				qs.append(q_t_log_prob)
				fs.append(f_t_log_prob)
				gs.append(g_t_log_prob)
				log_Ws.append(log_W)
				Ws.append(W)

				# no need to resample for t = time - 1
				if t == time - 1:
					break

				log_W = tf.transpose(log_W)
				categorical = tfd.Categorical(logits = log_W, validate_args=True, 
											  name = "Categorical_{}".format(t))
				if self.use_stop_gradient:
					idx = tf.stop_gradient(categorical.sample(self.n_particles))	# (n_particles, batch_size)
				else:
					idx = categorical.sample(self.n_particles)

				# ugly stuff used to resample X
				batch_1xB = tf.expand_dims(tf.range(batch_size), axis = 0)		# (1, batch_size)
				batch_NxB = tf.tile(batch_1xB, (self.n_particles, 1))			# (n_particles, batch_size)
				
				idx_NxBx1 = tf.expand_dims(idx, axis = 2)						# (n_particles, batch_size, 1)
				batch_NxBx1 = tf.expand_dims(batch_NxB, axis = 2)				# (n_particles, batch_size, 1)

				# print("emmm")
				# print(log_W)
				# print(idx)
				# print(batch_1xB)
				# print(batch_NxB)
				# print(idx_NxBx1)
				# print(batch_NxBx1)

				final_idx_NxBx2 = tf.concat((idx_NxBx1, batch_NxBx1), axis = 2)	# (n_particles, batch_size, 2)
				X_prev = tf.gather_nd(X, final_idx_NxBx2)						# (n_particles, batch_size, Dx)
				
				# collect X after rather than before resampling
				Xs.append(X_prev)

			# to make sure len(Xs) = time
			Xs.append(X)

			qs = tf.stack(qs, name = "qs")
			fs = tf.stack(fs, name = "fs")
			gs = tf.stack(gs, name = "gs")
			log_Ws = tf.stack(log_Ws, name = "log_Ws")
			Ws = tf.stack(Ws, name = "Ws")
			Xs = tf.stack(Xs, name = "Xs")

			mean_log_ZSMC = tf.reduce_mean(log_ZSMC, name = "mean_log_ZSMC")

		return mean_log_ZSMC, [Xs, log_Ws, Ws, fs, gs, qs]


	def n_step_MSE(self, n_steps, hidden, obs):
		"""
		compute MSE = (y_hat-y)^2 for n_steps
		for each x_t, calculate y_t_hat, y_t+1_hat, ..., y_t+n-1_hat
		then calculate the MSE between y_t:t+n-1_hat and y_t:t+n-1 as MSE_t
		finally, calculate the average of MSE_t for t in 0, 1, ..., T-n
		"""
		batch_size, time, Dx = hidden.shape.as_list()
		batch_size, time, Dy = obs.shape.as_list()
		with tf.variable_scope(self.name):

			# T = time - n_steps here
			x_BxTxDz = hidden[:, 0:time - n_steps + 1]

			# get y_hat
			ys_hat_BxNxTxDy = []

			for i in range(n_steps - 1):
				y_hat_BxTxD = self.g.mean(x_BxTxDz)
				ys_hat_BxNxTxDy.append(y_hat_BxTxD)
				x_BxTxD = self.f.mean(x_BxTxDz)

			y_hat_BxTxD = self.g.mean(x_BxTxDz)
			ys_hat_BxNxTxDy.append(y_hat_BxTxD)

			ys_hat_BxNxTxDy = tf.stack(ys_hat_BxNxTxDy, axis = 1)

			# get y_true
			ys_BxNxTxDy = []
			for t in range(time - n_steps + 1):
				ys_BxNxDy = obs[:, t:t+n_steps]
				ys_BxNxTxDy.append(ys_BxNxDy)
			ys_BxNxTxDy = tf.stack(ys_BxNxTxDy, axis = 2)

			# get MSE between y_hat and y_true
			MSE = tf.reduce_mean((ys_hat_BxNxTxDy - ys_BxNxTxDy)**2, name = "MSE")
			return MSE, ys_hat_BxNxTxDy, ys_BxNxTxDy
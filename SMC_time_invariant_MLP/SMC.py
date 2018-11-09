import tensorflow as tf
from tensorflow_probability import distributions as tfd

class SMC:
	def __init__(self, q, f, g,
				 n_particles, batch_size, 
				 use_stop_gradient = False,
				 name = "log_ZSMC"):
		self.q = q
		self.f = f
		self.g = g
		self.n_particles = n_particles
		self.batch_size = batch_size
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
		with tf.name_scope(self.name):
			batch_size, time, Dy = obs.get_shape().as_list()
			batch_size, Dx = x_0.get_shape().as_list()

			Xs = []
			log_Ws = []
			Ws = []
			fs = []
			gs = []
			qs = []

			# time = 0
			if self.name == 'log_ZSMC_true':
				X = self.q.sample(None, name='X0')
				q_0_log_probs = self.q.log_prob(None, X, name = 'q_0_log_probs')
				f_0_log_probs = self.f.log_prob(None, X, name = 'f_0_log_probs')
			else:
				x_0_y_0 = tf.concat([x_0, obs[:, 0]], axis = -1)
				X, q_0_log_probs = self.q.sample_and_log_prob(x_0_y_0, name = 'q_0_log_probs')
				f_0_log_probs = self.f.log_prob(x_0, X, name = 'f_0_log_probs')
			g_0_log_probs = self.g.log_prob(X, obs[:,0], name = 'g_0_log_probs')
			
			log_W = tf.add(f_0_log_probs, g_0_log_probs - q_0_log_probs, name = 'log_W_0')
			W = tf.exp(log_W, name = 'W_0')
			log_ZSMC = tf.log(tf.reduce_mean(W, axis = 0, name = 'W_0_mean'), name = 'log_ZSMC_0')

			log_Ws.append(log_W)
			Ws.append(W)
			fs.append(f_0_log_probs)
			gs.append(g_0_log_probs)
			qs.append(q_0_log_probs)

			for t in range(1, time):
				log_W = tf.transpose(log_W)
				categorical = tfd.Categorical(logits = log_W, validate_args=True, 
											  name = 'Categorical_{}'.format(t))
				if self.use_stop_gradient:
					idx = tf.stop_gradient(categoriucal.sample(self.n_particles))	# (n_particles, batch_size)
				else:
					idx = categorical.sample(self.n_particles)

				# ugly stuff used to resample X
				ugly_stuff = tf.tile(tf.expand_dims(tf.range(batch_size), axis = 0), (self.n_particles, 1)) 	# (n_particles, batch_size)
				idx_expanded = tf.expand_dims(idx, axis = 2)											# (n_particles, batch_size, 1)
				ugly_expanded = tf.expand_dims(ugly_stuff, axis = 2)									# (n_particles, batch_size, 1)
				final_idx = tf.concat((idx_expanded, ugly_expanded), axis = 2)							# (n_particles, batch_size, 2)
				X_prev = tf.gather_nd(X, final_idx)														# (n_particles, batch_size, Dx)
				
				# change Xs to collect X after rather than before resampling
				Xs.append(X_prev)

				# (n_particles, batch_size, Dx)
				if self.name == 'log_ZSMC_true':
					X = self.q.sample(X_prev, name = 'q_{}_sample'.format(t))
					q_t_log_probs = self.q.log_prob(X_prev, X, name = 'q_{}_log_probs'.format(t))
				else:
					y_t_expanded = tf.tile(tf.expand_dims(obs[:, t], axis = 0), (self.n_particles, 1, 1))
					X_prev_y_t = tf.concat([X_prev, y_t_expanded], axis = -1)
					X, q_t_log_probs = self.q.sample_and_log_prob(X_prev_y_t, name = 'q_{}_log_probs'.format(t))
				f_t_log_probs = self.f.log_prob(X_prev, X, name = 'f_{}_log_probs'.format(t))
				g_t_log_probs = self.g.log_prob(X, obs[:,t], name = 'g_{}_log_probs'.format(t))

				log_W = tf.add(f_t_log_probs, g_t_log_probs - q_t_log_probs, name = 'log_W_{}'.format(t))
				W = tf.exp(log_W, name = 'W_{}'.format(t))
				log_ZSMC += tf.log(tf.reduce_mean(W, axis = 0, name = 'W_0_mean'), name = 'log_ZSMC_{}'.format(t))

				Ws.append(W)
				log_Ws.append(log_W)
				fs.append(f_t_log_probs)
				gs.append(g_t_log_probs)
				qs.append(q_t_log_probs)

			# to make sure len(Xs) = time
			Xs.append(X)


			Xs = tf.stack(Xs)
			Ws = tf.stack(Ws)
			log_Ws = tf.stack(log_Ws)
			fs = tf.stack(fs)
			gs = tf.stack(gs)
			qs = tf.stack(qs)

			mean_log_ZSMC = tf.reduce_mean(log_ZSMC)

		return mean_log_ZSMC, [Xs, log_Ws, Ws, fs, gs, qs]

	def tf_accuracy(self, sess, log_ZSMC, obs, obs_set, x_0, hidden_set):
		"""
		used for evaluating true_log_ZSMC, train_log_ZSMC, test_log_ZSMC
		"""
		accuracy = 0
		for i in range(0, len(obs_set), self.batch_size):
			log_ZSMC_val = sess.run(log_ZSMC, feed_dict = {obs:obs_set[i:i+self.batch_size], 
														   x_0:[hidden[0] for hidden in hidden_set[i:i+self.batch_size]]})
			# print(i, log_ZSMC_val)
			accuracy += log_ZSMC_val
		return accuracy/(len(obs_set)/self.batch_size)


	def n_step_y_MSE(self, n_steps, hidden, obs):
		"""
		compute MSE = (y_hat-y)^2 for n_steps
		for each x_t, calculate y_t_hat, y_t+1_hat, ..., y_t+n-1_hat
		then calculate the MSE between y_t:t+n-1_hat and y_t:t+n-1 as MSE_t
		finally, calculate the average of MSE_t for t in 0, 1, ..., T-n
		"""
		batch_size, time, Dx = hidden.shape.as_list()
		batch_size, time, Dy = obs.shape.as_list()
		with tf.name_scope(self.name):
			MSEs = []
			for t in range(time - n_steps + 1):
				x = hidden[:, t]
				ys_hat = []
				for i in range(n_steps - 1):
					y_hat = self.g.get_mean(x)
					ys_hat.append(y_hat)
					x = self.f.get_mean(x)
				y_hat = self.g.get_mean(x)
				ys_hat.append(y_hat)
				ys_hat = tf.stack(ys_hat, axis = 1)
				ys = obs[:, t:t+n_steps]
				MSE = tf.reduce_mean((ys_hat - ys)**2, name = 'MSE_{}'.format(t))
				MSEs.append(MSE)
			MSEs = tf.stack(MSEs, name = 'MSEs')
			MSE_mean = tf.reduce_mean(MSEs, name = 'MSE_mean')
			return MSE_mean

	def tf_MSE(self, sess, MSE_mean, hidden, hidden_set, obs, obs_set):
		MSE_means_val = 0
		for i in range(0, len(hidden_set), self.batch_size):
			MSE_mean_val = sess.run(MSE_mean, feed_dict = {obs:obs_set[i:i+self.batch_size], 
														   hidden:hidden_set[i:i+self.batch_size]})
			MSE_means_val += MSE_mean_val
		return MSE_means_val/(len(obs_set)/self.batch_size)
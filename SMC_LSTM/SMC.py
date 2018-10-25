import tensorflow as tf
from tensorflow_probability import distributions as tfd

class SMC:
	def __init__(self, VRNN_Cell, q = None, f = None, g = None, p = None, 
				 use_stop_gradient = False, use_log_prob = True, 
				 name = "get_log_ZSMC"):
		"""
		self.VRNN_Cell: an instance of VartiationalRNN
	  	"""
		self.VRNN_Cell = VRNN_Cell
		self.q = q
		self.f = f
		self.g = g
		self.p = p
		self.use_log_prob = use_log_prob
		self.use_stop_gradient = use_stop_gradient
		self.name = name

	def get_log_ZSMC(self, obs):
		"""
		Input:
			obs.shape = (batch_size, time, Dy)
		Output:
			log_ZSMC: shape = scalar
			log: stuff to debug
		"""
		with tf.name_scope(self.name):
			Dx = self.VRNN_Cell.Dx
			n_particles = self.VRNN_Cell.n_particles
			batch_size, time, Dy = obs.get_shape().as_list()

			Xs = []
			log_Ws = []
			Ws = []
			fs = []
			gs = []
			qs = []
			ps = []

			# time = 1
			self.VRNN_Cell.reset()

			self.VRNN_Cell.step(obs[:,0])
			if self.q is None:
				X = self.VRNN_Cell.get_q_sample() 							# X.shape = (n_particles, batch_size, Dx)
				q_uno_log_probs = self.VRNN_Cell.get_q_log_prob(X)			# probs.shape = (n_particles, batch_size)
			else:
				X = self.q.sample(None, name = 'X0')
				q_uno_log_probs = self.q.log_prob(None, X, name = 'q_uno_probs')

			if self.f is None:
				f_nu_log_probs  = self.VRNN_Cell.get_f_log_prob(X)
			else:
				f_nu_log_probs  = self.f.log_prob(None, X, name = 'f_nu_probs')

			if self.g is None:
				g_uno_log_probs = self.VRNN_Cell.get_g_log_prob(X, obs[:,0])
			else:
				g_uno_log_probs = self.g.log_prob(X, obs[:,0], name = 'g_uno_probs')
			
			log_W = tf.add(f_nu_log_probs, g_uno_log_probs - q_uno_log_probs, name = 'log_W_0')
			W = tf.exp(log_W, name = 'W_0')
			log_ZSMC = tf.log(tf.reduce_mean(W, axis = 0, name = 'W_0_mean'), name = 'log_ZSMC_0')

			Xs.append(X)
			log_Ws.append(log_W)
			Ws.append(W)
			fs.append(f_nu_log_probs)
			gs.append(g_uno_log_probs)
			qs.append(q_uno_log_probs)
			ps.append(tf.zeros((n_particles, batch_size)))

			for t in range(1, time):

				# W_{t-1} = W_{t-1} * p(y_t | X_{t-1})
				# k = p.posterior(X, obs[t], name = 'p_{}'.format(t))
				k = tf.ones((n_particles, batch_size), dtype = tf.float32, name = 'p_{}'.format(t))
				log_k = tf.log(k)
				log_W = log_W + log_k
				log_W = tf.transpose(log_W)
				categorical = tfd.Categorical(logits = log_W, validate_args=True, 
											  name = 'Categorical_{}'.format(t))
				if self.use_stop_gradient:
					idx = tf.stop_gradient(categoriucal.sample(n_particles))	# (n_particles, batch_size)
				else:
					idx = categorical.sample(n_particles)

				# ugly stuff used to resample X
				ugly_stuff = tf.tile(tf.expand_dims(tf.range(batch_size), axis = 0), (n_particles, 1)) 	# (n_particles, batch_size)
				idx_expanded = tf.expand_dims(idx, axis = 2)											# (n_particles, batch_size, 1)
				ugly_expanded = tf.expand_dims(ugly_stuff, axis = 2)									# (n_particles, batch_size, 1)
				final_idx = tf.concat((idx_expanded, ugly_expanded), axis = 2)							# (n_particles, batch_size, 2)
				X_prev = tf.gather_nd(X, final_idx)														# (n_particles, batch_size, Dx)
						
				self.VRNN_Cell.update_lstm(X_prev, obs[:,t-1])

				self.VRNN_Cell.step(obs[:,t])

				if self.q is None:
					X = self.VRNN_Cell.get_q_sample() 							# X.shape = (n_particles, batch_size, Dx)
					q_t_log_probs = self.VRNN_Cell.get_q_log_prob(X)			# probs.shape = (n_particles, batch_size)
				else:
					X = self.q.sample(X_prev, name = 'q_{}_sample'.format(t))
					q_t_log_probs = self.q.log_prob(X_prev, X, name = 'q_{}_log_probs'.format(t))

				if self.f is None:
					f_t_log_probs  = self.VRNN_Cell.get_f_log_prob(X)
				else:
					f_t_log_probs  = self.f.log_prob(X_prev, X, name = 'f_{}_log_probs'.format(t))

				if self.g is None:
					g_t_log_probs = self.VRNN_Cell.get_g_log_prob(X, obs[:,0])
				else:
					g_t_log_probs = self.g.log_prob(X, obs[:,0], name = 'g_{}_log_probs'.format(t))

				log_W = tf.add(f_t_log_probs - log_k, g_t_log_probs - q_t_log_probs, name = 'log_W_{}'.format(t))
				W = tf.exp(log_W, name = 'W_{}'.format(t))
				log_ZSMC = tf.log(tf.reduce_mean(W, axis = 0, name = 'W_0_mean'), name = 'log_ZSMC_{}'.format(t))

				Xs.append(X)
				Ws.append(W)
				log_Ws.append(log_W)
				fs.append(f_t_log_probs)
				gs.append(g_t_log_probs)
				qs.append(q_t_log_probs)
				ps.append(log_k)

			Xs = tf.stack(Xs)
			Ws = tf.stack(Ws)
			log_Ws = tf.stack(log_Ws)
			fs = tf.stack(fs)
			gs = tf.stack(gs)
			qs = tf.stack(qs)
			ps = tf.stack(ps)

			mean_log_ZSMC = tf.reduce_mean(log_ZSMC)

		return mean_log_ZSMC, [Xs, log_Ws, Ws, fs, gs, qs, ps]

	def tf_accuracy(self, obs_set, obs, log_ZSMC, sess, batch_size):
		"""
		used for evaluating true_log_ZSMC, train_log_ZSMC, test_log_ZSMC
		"""
		accuracy = 0
		for i in range(0, len(obs_set), batch_size):
			log_ZSMC_val = sess.run(log_ZSMC, feed_dict = {obs:obs_set[i:i+batch_size]})
			# print(i, log_ZSMC_val)
			accuracy += log_ZSMC_val
		return accuracy/(len(obs_set)/batch_size)
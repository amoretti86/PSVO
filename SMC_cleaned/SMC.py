import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

class SMC:
	def __init__(self, n_particles, 
				 q, f, g, p, 
				 use_stop_gradient = False, use_log_prob = True, 
				 name = "get_log_ZSMC"):
		self.n_particles = n_particles
		self.q = q
		self.f = f
		self.g = g
		self.q = q
		self.use_stop_gradient = use_stop_gradient
		self.use_log_prob = use_log_prob
		self.name = name

	def get_log_ZSMC(self, obs):
		with tf.name_scope(self.name):

			T = obs.get_shape().as_list()[0]
			
			Xs = []
			log_Ws = []
			Ws = []
			fs = []
			gs = []
			qs = []
			ps = []

			# T = 1
			X = self.q.sample(None, name = 'X0')
			q_uno_log_probs = self.q.log_prob(None, X, name = 'q_uno_log_probs')
			f_nu_log_probs  = self.f.log_prob(None, X, name = 'f_nu_log_probs')
			g_uno_log_probs = self.g.log_prob(X, obs[0], name = 'g_uno_log_probs')

			log_W = tf.add(g_uno_log_probs, f_nu_log_probs - q_uno_log_probs, name = 'log_W_0')
			W = tf.exp(log_W, name = 'W_0')
			log_ZSMC = tf.log(tf.reduce_mean(W, name = 'W_0_mean'), name = 'log_ZSMC_0')

			Xs.append(X)
			log_Ws.append(log_W)
			Ws.append(W)
			fs.append(f_nu_log_probs)
			gs.append(g_uno_log_probs)
			qs.append(q_uno_log_probs)
			ps.append(tf.zeros(self.n_particles))

			for t in range(1, T):

				# log_W_{t-1} = log_W_{t-1} + log(self.p(y_t | X_{t-1}))
				# k = self.p.posterior(X, obs[t], name = 'p_{}'.format(t))
				k = tf.ones(self.n_particles, dtype = tf.float32)
				log_W = log_W + tf.log(k)

				categorical = tfd.Categorical(logits = log_W, validate_args=True, name = 'Categorical_{}'.format(t))
				if self.use_stop_gradient:
					idx = tf.stop_gradient(categorical.sample(self.n_particles))
				else:
					idx = categorical.sample(self.n_particles)

				X_prev = tf.gather(X, idx, validate_indices = True)
			
				X = self.q.sample(X_prev, name = 'q_{}_sample'.format(t))
				q_t_log_probs = self.q.log_prob(X_prev, X, name = 'q_{}_log_probs'.format(t))
				f_t_log_probs = self.f.log_prob(X_prev, X, name = 'f_{}_log_probs'.format(t))
				g_t_log_probs = self.g.log_prob(X, obs[t], name = 'g_{}_log_probs'.format(t))

				log_W = tf.add(g_t_log_probs - k, f_t_log_probs - q_t_log_probs, name = 'log_W_{}'.format(t))
				W = tf.exp(log_W, name = 'W_{}'.format(t))
				log_ZSMC += tf.log(tf.reduce_mean(W), name = 'log_ZSMC_{}'.format(t))

				Xs.append(X)
				log_Ws.append(log_W)
				Ws.append(W)
				fs.append(f_t_log_probs)
				gs.append(g_t_log_probs)
				qs.append(q_t_log_probs)
				ps.append(k)

			Xs = tf.stack(Xs, name = 'Xs')
			log_Ws = tf.stack(log_Ws, name = 'log_Ws')
			Ws = tf.stack(Ws, name = 'Ws')
			fs = tf.stack(fs, name = 'fs')
			gs = tf.stack(gs, name = 'gs')
			qs = tf.stack(qs, name = 'qs')
			ps = tf.stack(ps, name = 'ps')

			return log_ZSMC, [Xs, log_Ws, Ws, fs, gs, qs, ps]

	def tf_accuracy(self, obs_set, obs_placeholder, log_ZSMC, sess):
		"""
		used for evaluating true_log_ZSMC, train_log_ZSMC, test_log_ZSMC
		"""
		accuracy = 0
		for i, obs in enumerate(obs_set):
			log_ZSMC_val = sess.run(log_ZSMC, feed_dict = {obs_placeholder:obs})
			# print(i, log_ZSMC_val)
			accuracy += log_ZSMC_val
		return accuracy/len(obs_set)


""" get_log_ZSMC doesn't use log_prob

	def get_log_ZSMC(self, obs):
		with tf.name_scope(self.name):

			T = obs.get_shape().as_list()[0]
			
			Xs = []
			Ws = []
			W_means = []
			fs = []
			gs = []
			qs = []
			ps = []

			# T = 1
			X = self.q.sample(None, name = 'X0')
			q_uno_probs = self.q.prob(None, X, name = 'q_uno_probs')
			f_nu_probs  = self.f.prob(None, X, name = 'f_nu_probs')
			g_uno_probs = self.g.prob(X, obs[0], name = 'g_uno_probs')

			W = tf.multiply(g_uno_probs, f_nu_probs / q_uno_probs, name = 'W_0')
			log_ZSMC = tf.log(tf.reduce_mean(W, name = 'W_0_mean'), name = 'log_ZSMC_0')

			Xs.append(X)
			Ws.append(W)
			W_means.append(tf.reduce_mean(W))
			fs.append(f_nu_probs)
			gs.append(g_uno_probs)
			qs.append(q_uno_probs)
			ps.append(tf.zeros(self.n_particles))

			for t in range(1, T):

				# W_{t-1} = W_{t-1} * self.p(y_t | X_{t-1})
				# k = self.p.posterior(X, obs[t], name = 'p_{}'.format(t))
				k = tf.ones(self.n_particles, dtype = tf.float32)
				W = W * k

				categorical = tfd.Categorical(probs = W/tf.reduce_sum(W), name = 'Categorical_{}'.format(t))
				if self.use_stop_gradient:
					idx = tf.stop_gradient(categorical.sample(self.n_particles))
				else:
					idx = categorical.sample(self.n_particles)

				X_prev = tf.gather(X, idx, validate_indices = True)
			
				X = self.q.sample(X_prev, name = 'q_{}_sample'.format(t))
				q_t_probs = self.q.prob(X_prev, X, name = 'q_{}_probs'.format(t))
				f_t_probs = self.f.prob(X_prev, X, name = 'f_{}_probs'.format(t))
				g_t_probs = self.g.prob(X, obs[t], name = 'g_{}_probs'.format(t))

				W =  tf.divide(g_t_probs * f_t_probs, k * q_t_probs, name = 'W_{}'.format(t))
				log_ZSMC += tf.log(tf.reduce_mean(W), name = 'log_ZSMC_{}'.format(t))

				Xs.append(X)
				Ws.append(W)
				W_means.append(tf.reduce_mean(W))
				fs.append(f_t_probs)
				gs.append(g_t_probs)
				qs.append(q_t_probs)
				ps.append(k)

			Xs = tf.stack(Xs, name = 'Xs')
			Ws = tf.stack(Ws, name = 'Ws')
			fs = tf.stack(fs, name = 'fs')
			gs = tf.stack(gs, name = 'gs')
			qs = tf.stack(qs, name = 'qs')
			ps = tf.stack(ps, name = 'ps')
			W_means = tf.stack(W_means, name = 'W_means')

			return log_ZSMC, [Xs, Ws, fs, gs, qs, ps, W_means]
"""
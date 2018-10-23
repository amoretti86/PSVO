import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np

def get_log_ZSMC(obs, n_particles, q, f, g, p, use_stop_gradient, name = "get_log_ZSMC"):
	with tf.name_scope(name):

		T = obs.get_shape().as_list()[0]
		
		Xs = []
		Ws = []
		W_means = []
		fs = []
		gs = []
		qs = []
		ps = []

		# T = 1
		X = q.sample(None, name = 'X0')
		q_uno_probs = q.prob(None, X, name = 'q_uno_probs')
		f_nu_probs  = f.prob(None, X, name = 'f_nu_probs')
		g_uno_probs = g.prob(X, obs[0], name = 'g_uno_probs')

		W = tf.multiply(g_uno_probs, f_nu_probs / q_uno_probs, name = 'W_0')
		log_ZSMC = tf.log(tf.reduce_mean(W, name = 'W_0_mean'), name = 'log_ZSMC_0')

		Xs.append(X)
		Ws.append(W)
		W_means.append(tf.reduce_mean(W))
		fs.append(f_nu_probs)
		gs.append(g_uno_probs)
		qs.append(q_uno_probs)
		ps.append(tf.zeros(n_particles))

		for t in range(1, T):

			# W_{t-1} = W_{t-1} * p(y_t | X_{t-1})
			# k = p.posterior(X, obs[t], name = 'p_{}'.format(t))
			k = tf.ones(n_particles, dtype = tf.float32)
			W = W * k

			categorical = tfd.Categorical(probs = W/tf.reduce_sum(W), name = 'Categorical_{}'.format(t))
			if use_stop_gradient:
				idx = tf.stop_gradient(categorical.sample(n_particles))
			else:
				idx = categorical.sample(n_particles)

			X_prev = tf.gather(X, idx, validate_indices = True)
		
			X = q.sample(X_prev, name = 'q_{}_sample'.format(t))
			q_t_probs = q.prob(X_prev, X, name = 'q_{}_probs'.format(t))
			f_t_probs = f.prob(X_prev, X, name = 'f_{}_probs'.format(t))
			g_t_probs = g.prob(X, obs[t], name = 'g_{}_probs'.format(t))

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

def evaluate_mean_log_ZSMC(log_ZSMC, obs_samples, sess, obs):
	"""
	used for evaluating true_log_ZSMC, train_log_ZSMC, test_log_ZSMC
	"""
	log_ZSMCs = []
	for obs_sample in obs_samples:
		log_ZSMC_val = sess.run(log_ZSMC, feed_dict={obs: obs_sample})
		log_ZSMCs.append(log_ZSMC_val)

	return np.mean(log_ZSMCs)
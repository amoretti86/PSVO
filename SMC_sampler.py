import numpy as np
import math

class SMC_sampler:
	def __init__(self, Dx, Dy):
		self.Dx = Dx
		self.Dy = Dy

	def makePLDS(self, T, x_0, f, g):
		"""
		creates PLDS with exp link function
		T
		x_0
		f: transition class with sample() and prob():
			f.sample(x_t-1) will return a sample of x_t based on x_t-1
			f.prob(x_t-1, x_t) will return the probability f(x_t | x_t-1)
			to get prob nu(x_1) at T = 1, use f.prob(None, x_1)
		g: emission class with y_t = g.sample(x_t) and prob = g.prob(x_t, y_t)
		"""
		assert len(x_0) == self.Dx, "x_0 has wrong dim {} rather than ({}, )".format(x_0.shape, self.Dx)

		X = np.zeros((T, self.Dx))
		Y = np.zeros((T, self.Dy))

		X[0] = x_0
		Y[0] = g.sample(x_0)
		for t in range(1,T):
			X[t] = f.sample(X[t-1])
			Y[t] = g.sample(X[t])
		return X, Y

	def log_W_to_W(self, log_W):
		return np.exp(log_W - np.max(log_W))

	def sample(self, obs, n_particles, q, f, g, p, use_log_prob = False):
		"""
		obs.shape = (T, Dy)

		f: transition class with sample() and prob():
			f.sample(x_t-1) will return a sample of x_t based on x_t-1
			f.prob(x_t-1, x_t) will return the probability f(x_t | x_t-1)
			to get prob nu(x_1) at T = 1, use f.prob(None, x_1)
		g: emission class with y_t = g.sample(x_t) and prob = g.prob(x_t, y_t)
		q: proposal class with x_t = q.sample(x_t-1) and prob = q.prob(x_t-1, x_t)
			to sample and get prob at T = 1, use x_1 = q.sample(None) and prob = q.prob(None, x_1)
		p: p.posterior(x_t-1, y_t) returns \int f(x_t | x_t-1) g(y_t | x_t) dx_t
		"""
		T, Dy = obs.shape
		assert Dy == self.Dy, "obs has wrong dim {} rather than ({}, {})".format(obs.shape, T, self.Dy)

		# Initialize variables
		X = np.zeros((n_particles, T, self.Dx))
		a = np.zeros((n_particles, T))
		W = np.zeros((n_particles, T))
		log_W = np.zeros((n_particles, T))
		k = np.zeros((n_particles, T))
		log_k = np.zeros((n_particles, T))

		# Initialize particles and weights at T 1
		for i in range(n_particles):
			X[i, 0, :] = q.sample(None)
			if use_log_prob:
				log_nu = f.log_prob(None, X[i, 0, :])
				log_g = g.log_prob(X[i, 0, :], obs[0])
				log_q = q.log_prob(None, X[i, 0, :])
				log_W[i, 0] = log_g + log_nu - log_q
			else:
				nu_prob = f.prob(None, X[i, 0, :])
				g_prob = g.prob(X[i, 0, :], obs[0])
				q_prob = q.prob(None, X[i, 0, :])
				log_W[i, 0] = g_prob * nu_prob / q_prob

		if use_log_prob:
			W[:, 0] = self.log_W_to_W(log_W[:, 0])

		# Define recursive approximation to p(z_{1:n}|x_{1:n})
		for t in range(1, T):

			# Update weights and propagate particles based on posterior integral
			for i in range(n_particles):
				if use_log_prob:
					log_k[i, t] = p.log_posterior(X[i, t - 1, :], obs[t, :])
					# print("Laplace particle {}, mass: {}".format(i, k1[i, t]))
					assert(math.isfinite(log_k[i, t])), 'LaplaceApprox generates invalid number {}'.format(log_k[i, t])
					# Reweight particles
					log_W[i, t - 1] = log_W[i, t - 1] + log_k[i, t]
				else:
					k[i, t] = p.posterior(X[i, t - 1, :], obs[t, :])
					# print("Laplace particle {}, mass: {}".format(i, k1[i, t]))
					assert(math.isfinite(k[i, t])), 'LaplaceApprox generates invalid number {}'.format(k[i, t])
					# Reweight particles
					W[i, t - 1] = W[i, t - 1] * k[i, t]

			if use_log_prob:
				W[:, t - 1] = self.log_W_to_W(log_W[:, t - 1])

			# normalize W
			W[:, t - 1] = W[:, t - 1] / np.sum(W[:, t - 1])

			# Resample
			Xprime = np.random.choice(n_particles, n_particles, p = W[:, t - 1], replace = True)
			a[:, t] = Xprime
			Xtilde = [X[i, t - 1, :] for i in Xprime]

			# Reset weights and particles
			for i in range(n_particles):
				# Resample particles and reset weights
				X[i, t, :] = q.sample(Xtilde[i])

				# Update factorized proposal and target distributions
				if use_log_prob:
					log_f = f.log_prob(Xtilde[i], X[i, t, :])
					log_g = g.log_prob(X[i, t, :], obs[t])
					log_q = q.log_prob(Xtilde[i], X[i, t, :])
					# Update weights
					log_W[i, t] = log_g + log_f - log_k[i, t] - log_q
				else:
					f_prob = f.prob(Xtilde[i], X[i, t, :])
					g_prob = g.prob(X[i, t, :], obs[t])
					q_prob = q.prob(Xtilde[i], X[i, t, :])
					# Update weights
					W[i, t] = (g_prob * f_prob) / (k[i, t] * q_prob)

			if use_log_prob:
				W[:, t] = self.log_W_to_W(log_W[:, t])

		X = X.astype(np.float32)
		return W, X, k, a
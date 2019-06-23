
import scipy as sp
import numpy as np

import tensorflow as tf
import tensorflow.contrib.distributions as tfd

class LaplaceApprox:
	"""
	LaplaceApprox to compute posterior approx: \int f(z_t | z_t-1) g(y_t | z_t) dz_t
	where f(z_t | z_t-1) is multivariate_normal
		  g(y_t | z_t)   is element-wise poisson
	"""
	def __init__(self, A_DzxDz, Q_DzxDz, B_DyxDz, n_iters=2):
		"""
		z_t ~ multivariate_normal(A * z_t-1, Q)
		y_t ~ poisson(B * z_t)
		"""
		self.A_DzxDz = A_DzxDz
		self.Q_DzxDz = Q_DzxDz
		self.B_DyxDz = B_DyxDz
		self.n_iters = n_iters

		# pre-calculated to speed up posterior calculation
		self.Q_inv_DzxDz = np.linalg.inv(Q_DzxDz)

	def get_Z_and_H(self, Zprev_Dz, y_Dy, debug_mode=False):
		Z_Dz = Zprev_Dz
		mu_Dz = np.dot(self.A_DzxDz, Zprev_Dz)

		BtY_Dz = np.dot(self.B_DyxDz.T, y_Dy)
		QBtY_Dz = np.dot(self.Q_DzxDz, BtY_Dz)

		for i in range(self.n_iters):
			# Compute FPI for the mean
			BZ_Dy = np.dot(self.B_DyxDz, Z_Dz)
			expBZ_Dy = np.exp(BZ_Dy)
			BtexpBZ_Dz = np.dot(self.B_DyxDz.T, expBZ_Dy)

			Z_Dz = - np.dot(self.Q_DzxDz, BtexpBZ_Dz) + QBtY_Dz + mu_Dz
			if debug_mode:
				print("iter %i, mean:\n"%i, Z_Dz)

		# Compute FPI for the hessian
		expBZ_DyxDy = np.diag(np.exp(np.dot(self.B_DyxDz, Z_Dz)))
		BtexpBZ_DzxDy = np.dot(self.B_DyxDz.T, expBZ_DyxDy)
		BtexpBZB_DzxDz = np.dot(BtexpBZ_DzxDy, self.B_DyxDz)
		H_DzxDz = BtexpBZB_DzxDz + self.Q_inv_DzxDz
		if debug_mode:
			print("Hessian:\n" % i, H_DzxDz)

		return Z_Dz, H_DzxDz

	def posterior(self, Zprev_Dz, y_Dy):
		""" 
		Computes Laplace Approximation to 
		exp{ -1/2 (z_t - mu)^T Q^{-1}(z - mu) - <exp(Bz),1> + <Bz,y> - <ln(y!),1> }
		"""
		mu_Dz = np.dot(self.A_DzxDz, Zprev_Dz)
		Z_Dz, H_DzxDz = self.get_Z_and_H(Zprev_Dz, y_Dy)

		SqInvDet = (1 / np.linalg.det(H_DzxDz)) ** (1. / 2)
		PiTerm = (2 * np.pi) ** (self.Q_DzxDz.shape[0] / 2)
		Pstar = sp.stats.multivariate_normal.pdf(Z_Dz, mu_Dz, self.Q_DzxDz) \
				* np.prod(sp.stats.poisson.pmf(y_Dy, np.exp(np.dot(self.B_DyxDz, Z_Dz))))
		Ztilde_1x1 =  SqInvDet * PiTerm * Pstar
		# print "area: ", Ztilde_1x1

		return Ztilde_1x1

	def log_posterior(self, Zprev_Dz, y_Dy):
		""" 
		Computes Laplace Approximation to 
		-1/2 (z_t - mu)^T Q^{-1}(z - mu) - <exp(Bz),1> + <Bz,y> - <ln(y!),1>
		"""
		mu_Dz = np.dot(self.A_DzxDz, Zprev_Dz)
		Z_Dz, H_DzxDz = self.get_Z_and_H(Zprev_Dz, y_Dy)

		log_SqInvDet = (-1./2) * np.log(np.linalg.det(H_DzxDz))
		log_PiTerm = (self.Q_DzxDz.shape[0] / 2) * np.log(2 * np.pi)
		log_Pstar = sp.stats.multivariate_normal.logpdf(Z_Dz, mu_Dz, self.Q_DzxDz) \
					+ np.sum(sp.stats.poisson.logpmf(y_Dy, np.exp(np.dot(self.B_DyxDz, Z_Dz))))
		log_Ztilde_1x1 = log_SqInvDet + log_PiTerm + log_Pstar
		# print "area: ", log_Ztilde_1x1

		return log_Ztilde_1x1

class TensorLaplaceApprox:
	"""
	TesnorLaplaceApprox to compute posterior approx: \int f(z_t | z_t-1) g(y_t | z_t) dz_t
	where f(z_t | z_t-1) is multivariate_normal
		  g(y_t | z_t)   is element-wise poisson
	"""
	def __init__(self, A_DzxDz, Q_DzxDz, B_DyxDz, n_iters=2, name = 'TensorLaplaceApprox'):
		"""
		z_t ~ multivariate_normal(A * z_t-1, Q)
		y_t ~ poisson(B * z_t)
		"""
		self.name = name

		with tf.name_scope(self.name):
			self.A_DzxDz = tf.identity(A_DzxDz, name = 'A_DzxDz')
			self.Q_DzxDz = tf.identity(Q_DzxDz, name = 'Q_DzxDz')
			self.B_DyxDz = tf.identity(B_DyxDz, name = 'B_DyxDz')
			self.n_iters = n_iters
			# pre-calculated to speed up posterior calculation
			self.Q_inv_DzxDz = tf.matrix_inverse(Q_DzxDz, name = 'Q_inv_DzxDz')

	def posterior(self, Zprev_NxDz, y_Dy, name = None, debug_mode=False):
		""" Computes Laplace Approx using an FPI for the first and second moments differentiating the log posterior """
		if name is None:
			name = self.name
		with tf.name_scope(name):
			Dz = tf.constant(self.A_DzxDz.get_shape().as_list()[0], dtype = tf.float32)
			N = Zprev_NxDz.get_shape().as_list()[0]

			Z_NxDz = tf.identity(Zprev_NxDz, name = 'Z_NxDz')
			mu_NxDz = tf.matmul(Zprev_NxDz, self.A_DzxDz, transpose_b = True, name = 'mu_NxDz')

			QBt_DzxDy = tf.matmul(self.Q_DzxDz, self.B_DyxDz, transpose_b = True)
			QBtY_Dz = tf.matmul(tf.expand_dims(y_Dy, axis = 0), QBt_DzxDy, transpose_b = True, name = 'QBtY_Dz')

			# Iterate over FPIs for first and second moments:
			for i in range(self.n_iters):
				# Compute FPI for the mean
				BZ_NxDy = tf.matmul(Z_NxDz, self.B_DyxDz, transpose_b = True, name = 'BZ_NxDy')
				expBZ_NxDy = tf.exp(BZ_NxDy, name = 'expBZ_NxDy')
				BtexpBZ_NxDz = tf.matmul(expBZ_NxDy, self.B_DyxDz, name = 'BtexpBZ_Dz')

				Z_NxDz = - tf.matmul(BtexpBZ_NxDz, self.Q_DzxDz, transpose_b = True) + QBtY_Dz + mu_NxDz
				if debug_mode:
					print("iter %i, mean:\n"%i, Z_NxDz)

			# Compute FPI for the Hessian
			expBZ_NxDyxDy = tf.matrix_diag(tf.exp(tf.matmul(Z_NxDz, self.B_DyxDz, transpose_b=True)), name = 'expBZ_NxDyxDy')
			BtexpBZ_NxDzxDy = tf.einsum('ijk,jh->ihk', expBZ_NxDyxDy, self.B_DyxDz)
			BtexpBZB_NxDzxDz = tf.einsum('ijk,kh->ijh', BtexpBZ_NxDzxDy, self.B_DyxDz)
			H_NxDzxDz = BtexpBZB_NxDzxDz + self.Q_inv_DzxDz
			if debug_mode:
				print("Tensor of Hessians:\n", H_NxDzxDz)

			# Compute the inverse normalization to approximate the integral
			SqInvDet = 1./tf.sqrt(tf.matrix_determinant(H_NxDzxDz))
			PiTerm = (2*tf.constant(np.pi))**(Dz/2)

			mvn_loc = tf.matmul(Zprev_NxDz, self.A_DzxDz, transpose_b = True, name = 'mvn_loc')
			mvn = tfd.MultivariateNormalFullCovariance(loc = mvn_loc, 
													   covariance_matrix = self.Q_DzxDz,
													   name = "mvn")
			mvn_prob = mvn.prob(Z_NxDz, name = "mvn_prob")

			log_rate = tf.matmul(Z_NxDz, self.B_DyxDz, transpose_b = True, name = 'log_rate')
			poisson = tfd.Poisson(log_rate = log_rate, name = "Poisson")
			y_NxDy = tf.tile(tf.expand_dims(y_Dy, axis = 0), [N, 1], name = 'y_NxDy')
			element_wise_prob = poisson.prob(y_NxDy, name = "element_wise_prob")
			poisson_prob = tf.reduce_sum(element_wise_prob, axis = 1, name = "poisson_prob")

			Pstar = mvn_prob * poisson_prob

			Ztilde_Nx1 = SqInvDet * PiTerm * Pstar
			return Ztilde_Nx1

class GaussianPostApprox():
	"""
	compute posterior approx: \int f(z_t | z_t-1) g(y_t | z_t) dz_t
	where f(z_t | z_t-1) is multivariate_normal
		  g(y_t | z_t)   is multivariate_normal
	check http://compbio.fmph.uniba.sk/vyuka/ml/old/2008/handouts/matrix-cookbook.pdf section 8.1.8
	"""
	def __init__(self, A_DzxDz, B_DyxDz, Q_DzxDz, Sigma_DyxDy):
		"""
		z_t ~ multivariate_normal(A * z_t-1, Q)
		y_t ~ multivariate_normal(B * z_t, Sigma)
		"""
		self.A_DzxDz = A_DzxDz
		self.B_DyxDz = B_DyxDz
		self.Q_DzxDz = Q_DzxDz
		self.Sigma_DyxDy = Sigma_DyxDy

		# pre-calculated to speed up posterior calculation
		BtB_DzxDz = np.dot(B_DyxDz.T, B_DyxDz)
		self.B_inv_DzxDy = np.dot(np.linalg.inv(BtB_DzxDz), B_DyxDz.T)

		BBt_DyxDy = np.dot(B_DyxDz, B_DyxDz.T)
		BBt_inv_DyxDy = np.linalg.inv(BBt_DyxDy)
		B_inv_inv_DyxDz = np.dot(BBt_inv_DyxDy, np.dot(B_DyxDz, BtB_DzxDz))     # inv of B_inv

		Sigma_inv_DyxDy = np.linalg.inv(Sigma_DyxDy)
		Sigma2_DzxDz = np.linalg.inv(np.dot(B_inv_inv_DyxDz.T, np.dot(Sigma_inv_DyxDy, B_inv_inv_DyxDz)))

		self.Sigma_sum_DzxDz = Q_DzxDz + Sigma2_DzxDz
		self.Sigma_sum_inv_DzxDz = np.linalg.inv(self.Sigma_sum_DzxDz)

	def posterior(self, Zprev_Dz, y_Dy):
		Dz = self.A_DzxDz.shape[0]

		mu1_Dz = np.dot(self.A_DzxDz, Zprev_Dz)
		mu2_Dz = np.dot(self.B_inv_DzxDy, y_Dy)
		mu_diff_Dz = mu1_Dz - mu2_Dz

		det_term = 1/np.sqrt((np.pi)**Dz * np.linalg.det(self.Sigma_sum_DzxDz))
		exp_term = np.exp(-1./2 * np.dot(mu_diff_Dz.T, np.dot(self.Sigma_sum_inv_DzxDz, mu_diff_Dz)))

		k = det_term * exp_term

		return k

	def log_posterior(self, Zprev_Dz, y_Dy):
		Dz = self.A_DzxDz.shape[0]

		mu1_Dz = np.dot(self.A_DzxDz, Zprev_Dz)
		mu2_Dz = np.dot(self.B_inv_DzxDy, y_Dy)
		mu_diff_Dz = mu1_Dz - mu2_Dz

		log_det_term = -(1./2) * (Dz * np.log(2 * np.pi) + np.log(np.linalg.det(self.Sigma_sum_DzxDz)))
		log_exp_term = -1./2 * np.dot(mu_diff_Dz.T, np.dot(self.Sigma_sum_inv_DzxDz, mu_diff_Dz))

		log_k = log_det_term + log_exp_term

		return log_k

class TensorGaussianPostApprox():
	"""
	compute posterior approx: \int f(z_t | z_t-1) g(y_t | z_t) dz_t
	where f(z_t | z_t-1) is multivariate_normal
		  g(y_t | z_t)   is multivariate_normal
	check http://compbio.fmph.uniba.sk/vyuka/ml/old/2008/handouts/matrix-cookbook.pdf section 8.1.8
	"""
	def __init__(self, A_DzxDz, B_DyxDz, Q_DzxDz, Sigma_DyxDy, name = 'TensorGaussianPostApprox'):
		"""
		z_t ~ multivariate_normal(A * z_t-1, Q)
		y_t ~ multivariate_normal(B * z_t, Sigma)
		"""
		self.name = name

		with tf.name_scope(self.name):
			self.A_DzxDz = tf.identity(A_DzxDz, name = 'A_DzxDz')
			self.B_DyxDz = tf.identity(B_DyxDz, name = 'B_DyxDz')
			self.Q_DzxDz = tf.identity(Q_DzxDz, name = 'Q_DzxDz')
			self.Sigma_DyxDy = tf.identity(Sigma_DyxDy, name = 'Sigma_DyxDy')

			BtB_DzxDz = tf.matmul(tf.transpose(B_DyxDz), B_DyxDz, name = 'BtB_DzxDz')
			self.B_inv_DzxDy = tf.matmul(tf.matrix_inverse(BtB_DzxDz), tf.transpose(B_DyxDz), name = 'B_inv_DzxDy')

			BBt_DyxDy = tf.matmul(B_DyxDz, tf.transpose(B_DyxDz), name = 'BBt_DyxDy')
			BBt_inv_DyxDy = tf.matrix_inverse(BBt_DyxDy, name = 'BBt_inv_DyxDy')
			# inv of B_inv
			B_inv_inv_DyxDz = tf.matmul(BBt_inv_DyxDy, tf.matmul(B_DyxDz, BtB_DzxDz), name = 'B_inv_inv_DyxDz')     
			
			Sigma1_DzxDz = tf.identity(Q_DzxDz, name = 'Sigma1_DzxDz')
			Sigma_inv_DyxDy = tf.matrix_inverse(Sigma_DyxDy, name = 'Sigma_inv_DyxDy')
			Sigma2_DzxDz = tf.matrix_inverse(tf.matmul(tf.transpose(B_inv_inv_DyxDz), tf.matmul(Sigma_inv_DyxDy, B_inv_inv_DyxDz)), name = 'Sigma2_DzxDz')

			self.Sigma_sum_DzxDz = tf.add(Sigma1_DzxDz, Sigma2_DzxDz, name = 'Sigma_sum_DzxDz')
			self.Sigma_sum_inv_DzxDz = tf.matrix_inverse(self.Sigma_sum_DzxDz, name = 'Sigma_sum_inv_DzxDz')

	def posterior(self, X_prev_NxDz, y_Dy, name = None):
		if name is None:
			name = self.name 
		with tf.name_scope(name):
			Dz = tf.shape(self.A_DzxDz, out_type = tf.float32)[0]

			mu1_NxDz = tf.matmul(X_prev_NxDz, self.A_DzxDz, transpose_b = True)
			mu2_Dz = tf.matmul(tf.expand_dims(y_Dy, axis = 0), self.B_inv_DzxDy, transpose_b = True)
			mu_diff_NxDz = mu1_NxDz - mu2_Dz

			mu_diff_Sigma_sum_NxDz = tf.matmul(mu_diff_NxDz, self.Sigma_sum_inv_DzxDz, transpose_b = True)
			exp_term_N = tf.exp(-1./2 * tf.reduce_sum(mu_diff_NxDz * mu_diff_Sigma_sum_NxDz, axis = 1))
			det_term = 1./tf.sqrt((tf.constant(2 * np.pi))**Dz * tf.matrix_determinant(self.Sigma_sum_DzxDz))

			k_N = det_term * exp_term_N

		return k_N

	def batchPosterior(self, X_prev_NxMxDz, y_MxDy, name=None):
		""" Define computation of posterior probabilities to operate on minibatches
			accepting a 3-Tensor for X_prev_NxMxDz and a 2-Tensor for y_MxDy """
		if name is None:
			name = self.name
		with tf.name_scope(name):
			Dz = tf.shape(self.A_DzxDz, out_type = tf.float32)[0]

			mu1_NxMxDz = tf.einsum('nmz,zy->nmy',X_prev_NxMxDz,self.A_DzxDz)
			mu2_MxDz = tf.matmul(y_MxDy, self.B_inv_DzxDy, transpose_b = True)
			mu_diff_NxMxDz = tf.subtract(mu1_NxMxDz, mu2_MxDz)

			mu_diff_Sigma_sum_NxMxDz = tf.einsum('nmz,zy->nmy', mu_diff_NxMxDz, self.Sigma_sum_inv_DzxDz)
			exp_term_NxM = tf.exp(-(1./2)* tf.reduce_sum(mu_diff_NxMxDz * mu_diff_Sigma_sum_NxMxDz, axis=2))
			det_term = 1./tf.sqrt((tf.constant(2*np.pi))**Dz *tf.matrix_determinant(self.Sigma_sum_DzxDz))

			k_NxM = det_term * exp_term_NxM

		return k_NxM
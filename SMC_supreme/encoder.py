import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from transformation.base import transformation

class encoder_cell:
	def __init__(self, Dx, Dy, batch_size, time, name = "encoder_cell"):
		self.Dx = Dx
		self.Dy = Dy
		self.batch_size = batch_size
		self.time = time

		self.Enc_layer_1_dim = (self.time - 1) * 5
		self.Enc_layer_2_dim = (self.time - 1) * 5
		self.Ev_layer_1_dim = 200
		self.Ev_layer_2_dim = 200

		self.alpha = 0.1
		
		self.name = name

		self.q_transformation = self.list_linear_transformation(None)
		self.f_transformation = self.list_linear_transformation(None)
		self.g_transformation = self.list_linear_transformation(None)

	class list_linear_transformation(transformation):
		def update_L_A(self, list_A):
			self.list_A = list_A
			self.t = 0

		def transform(self, Input):
			"""
			Input.shape = [n_particles, batch_size, Dx]
			"""
			A = self.list_A[self.t]
			self.t += 1
			return tf.einsum("bjk,nbk->nbj", A, Input)

	# ================================ keep an eye on len of YInput ================================ #
	# =================================== len = time or time - 1 =================================== #
	def encoding(self, YInput_NbxTxDy):
		with tf.variable_scope(self.name + "/encoding"):
			YInput_reshape = tf.reshape(YInput_NbxTxDy, [self.batch_size, (self.time - 1) * self.Dy], name = "YInput_reshape")
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
			X_hat_flat = fully_connected(Enc_layer_2, (self.time - 1) * self.Dx,
											weights_initializer=tf.orthogonal_initializer(),
											biases_initializer=tf.zeros_initializer(),
											activation_fn=None,
											reuse = tf.AUTO_REUSE, scope = "X_hat_flat")
			X_hat_NbxTxDz = tf.reshape(X_hat_flat, [self.batch_size, self.time - 1, self.Dx], name = "X_hat_NbxTxDz")
			return X_hat_NbxTxDz

	def evolving(self, X_hat_NbxTxDz):
		with tf.variable_scope(self.name + "/evolving"):
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
			B_flat  = fully_connected(Ev_layer_2, self.Dx**2,
									  weights_initializer=tf.orthogonal_initializer(),
									  biases_initializer=tf.zeros_initializer(),
									  activation_fn=None,
									  reuse = tf.AUTO_REUSE, scope = "B_flat")
			# check dim! time or time - 1
			B_NbxTxDzxDz = tf.reshape(B_flat, [self.batch_size, self.time - 1, self.Dx, self.Dx])
			# B must be symmetric
			B_NbxTxDzxDz = tf.matrix_band_part(B_NbxTxDzxDz, 0, -1)
			B_NbxTxDzxDz = 0.5 * (B_NbxTxDzxDz + tf.transpose(B_NbxTxDzxDz, perm=[0, 1, 3, 2]))
			return B_NbxTxDzxDz

	def get_q_As(self, YInput_NbxTxDy):
		X_hat_NbxTxDz = self.encoding(YInput_NbxTxDy)
		B_NbxTxDzxDz = self.evolving(X_hat_NbxTxDz)
		with tf.variable_scope(self.name + "/get_q_As"):
			A_NbxTxDzxDz = tf.add(tf.eye(self.Dx), self.alpha * B_NbxTxDzxDz, name = "A_NxTxDzxDz")
			A_L_NbxDzxDz = tf.unstack(A_NbxTxDzxDz, axis = 1, name = "A_L_NbxDzxDz")
			return A_NbxTxDzxDz, A_L_NbxDzxDz

	def encode(self, YInput_NbxTxDy, x_0_NbxDz):
		_, q_A_L_NbxDzxDz = self.get_q_As(YInput_NbxTxDy)
		self.q_transformation.update_L_A(q_A_L_NbxDzxDz)
		self.f_transformation.update_L_A(q_A_L_NbxDzxDz)


		# Xs_L_NbxDz = [x_0_NbxDz]
		# with tf.variable_scope(self.name + "/get_Xs"):
		# 	X_NbxDz = x_0_NbxDz
		# 	for q_A_NbxDzxDz in self.q_A_L_NbxDzxDz:
		# 		X_NbxDz = tf.einsum("ijk,ik->ij", q_A_NbxDzxDz, X_NbxDz)
		# 		Xs_L_NbxDz.append(X_NbxDz)
		# 	Xs_NbxTxDz = tf.stack(Xs_L_NbxDz, axis = 1, name = "Xs_NbxTxDz")
		# self.f_A_L_NbxDzxDz = self.get_f_As(Xs_NbxTxDz)
		# self.g_B_L_NbxDyxDz = self.get_g_Bs(Xs_NbxTxDz)

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from transformation.base import transformation

class MLP_transformation(transformation):
	def __init__(self, Dhs, Dout, name = "MLP_transformation"):
		self.Dhs = Dhs
		self.Dout = Dout
		self.name = name

	def transform(self, Input):
		with tf.variable_scope(self.name):
			hidden = tf.identity(Input, name = "hidden_0")
			for i, Dh in enumerate(self.Dhs):
				hidden = fully_connected(hidden, Dh,
										 weights_initializer=xavier_initializer(uniform=False), 
										 biases_initializer=tf.constant_initializer(0),
										 activation_fn = tf.nn.relu,
										 reuse = tf.AUTO_REUSE, scope = "hidden_{}".format(i))
			output = fully_connected(hidden, self.Dout,
									 weights_initializer=xavier_initializer(uniform=False), 
									 biases_initializer=tf.constant_initializer(0),
									 activation_fn = None,
									 reuse = tf.AUTO_REUSE, scope = "output")
		return output
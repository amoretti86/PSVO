import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from transformation.base import transformation


class MLP_transformation(transformation):
    def __init__(self, Dhs, Dout,
                 use_residual=False,
                 output_cov=False,
                 diag_cov=True,
                 name="MLP_transformation"):
        self.Dhs = Dhs
        self.Dout = Dout
        self.name = name
        self.use_residual = use_residual
        self.output_cov = output_cov
        self.diag_cov = diag_cov

    def transform(self, Input):
        with tf.variable_scope(self.name):
            hidden = tf.identity(Input, name="hidden_0")
            for i, Dh in enumerate(self.Dhs):
                hidden = fully_connected(hidden, Dh,
                                         weights_initializer=xavier_initializer(uniform=False),
                                         biases_initializer=tf.constant_initializer(0),
                                         activation_fn=tf.nn.relu,
                                         reuse=tf.AUTO_REUSE, scope="hidden_{}".format(i))
            mu = fully_connected(hidden, self.Dout,
                                 weights_initializer=xavier_initializer(uniform=False),
                                 biases_initializer=tf.constant_initializer(0),
                                 activation_fn=None,
                                 reuse=tf.AUTO_REUSE, scope="output_mu")
            if self.use_residual:
                mu += Input

            cov = None
            if self.output_cov:
                if self.diag_cov:
                    cov = fully_connected(hidden, self.Dout,
                                          weights_initializer=xavier_initializer(uniform=False),
                                          biases_initializer=tf.constant_initializer(0.6),
                                          activation_fn=tf.nn.softplus,
                                          reuse=tf.AUTO_REUSE, scope="output_cov")
                else:
                    cov = fully_connected(hidden, self.Dout**2,
                                          weights_initializer=xavier_initializer(uniform=False),
                                          biases_initializer=tf.constant_initializer(0.6),
                                          activation_fn=tf.nn.softplus,
                                          reuse=tf.AUTO_REUSE, scope="output_cov")
                    batch_size = hidden.shape.as_list()[:-1]
                    cov = tf.reshape(cov, batch_size + [self.Dout, self.Dout])

                    # make sure cov matrix is symmetric
                    perm = list(range(len(cov.shape)))
                    perm[-2], perm[-1] = perm[-1], perm[-2]
                    cov = (cov + tf.transpose(cov, perm=perm)) / 2

                cov += 1e-6  # to resolve numerical issues

        return mu, cov

import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from distribution.base import distribution


# np ver, just used in sampler, so no need to implement log_prob
class mvn(distribution):
    # multivariate normal distribution

    def __init__(self, transformation, sigma):
        self.sigmaChol = np.linalg.cholesky(sigma)
        self.transformation = transformation

    def sample(self, Input):
        mu = self.transformation.transform(Input)
        return mu + np.dot(self.sigmaChol, np.random.randn(len(mu)))


# tf ver, used in calculate log_ZSMC
class tf_mvn(distribution):
    # multivariate normal distribution

    def __init__(self, transformation,
                 output_0=None,
                 sigma_init=5, sigma_min=1,
                 name='tf_mvn'):
        self.transformation = transformation
        self.output_0 = output_0
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.name = name

    def get_mvn(self, Input=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            sigma = None

            if Input is None:
                assert self.output_0 is not None, "output_0 is not initialized for {}!".format(self.name)
                mu = self.output_0
            else:
                mu = self.transformation.transform(Input)
                if isinstance(mu, tuple):
                    assert len(mu) == 2, "output of {} should contain 2 elements".format(self.transformation.name)
                    mu, sigma = mu

            Dout = mu.shape.as_list()[-1]
            sigma_con = tf.get_variable("sigma_con",
                                        shape=[Dout],
                                        dtype=tf.float32,
                                        initializer=tf.constant_initializer(self.sigma_init),
                                        trainable=True)
            sigma_con = tf.nn.softplus(sigma_con)
            sigma_con = tf.where(tf.is_nan(sigma_con), tf.zeros_like(sigma_con), sigma_con)
            sigma_con = tf.maximum(sigma_con, self.sigma_min)

            if sigma is None:
                mvn = tfd.MultivariateNormalDiag(mu, sigma_con,
                                                 validate_args=True,
                                                 allow_nan_stats=False)
            else:
                if len(sigma.shape.as_list()) == len(mu.shape.as_list()):
                    sigma = sigma_con + 0.1 * sigma
                    mvn = tfd.MultivariateNormalDiag(mu, sigma,
                                                     validate_args=True,
                                                     allow_nan_stats=False)
                else:
                    sigma = tf.diag(sigma_con) + 0.1 * sigma
                    # sigma_shape_len = len(sigma.shape.as_list())
                    # axis = list(range(sigma_shape_len))
                    # axis[-2], axis[-1] = axis[-1], axis[-2]
                    # sigma = (sigma + tf.transpose(sigma, perm=axis)) / 2
                    mvn = tfd.MultivariateNormalFullCovariance(mu, sigma,
                                                               validate_args=True,
                                                               allow_nan_stats=False)

            return mvn

    def sample_and_log_prob(self, Input, sample_shape=(), name=None):
        mvn = self.get_mvn(Input)
        with tf.variable_scope(name or self.name):
            sample = mvn.sample(sample_shape)
            log_prob = mvn.log_prob(sample)
            return sample, log_prob

    def log_prob(self, Input, output, name=None):
        mvn = self.get_mvn(Input)
        with tf.variable_scope(name or self.name):
            return mvn.log_prob(output)

    def mean(self, Input, name=None):
        mvn = self.get_mvn(Input)
        with tf.variable_scope(name or self.name):
            return mvn.mean()

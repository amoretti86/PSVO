import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from transformation.flow import NF
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
                 sigma_init=5, sigma_min=1,
                 name='tf_mvn'):
        self.transformation = transformation
        self.sigma_init = sigma_init
        self.sigma_min = sigma_min
        self.name = name

    def get_mvn(self, Input):
        if isinstance(self.transformation, NF):
            dist = self.get_mvn_from_flow(Input)
        else:
            dist = self.get_mvn_from_transformation(Input)
        return dist

    def get_mvn_from_flow(self, Input):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            assert Input.shape.as_list()[-1] == self.transformation.event_size
            sigma = self.get_sigma(Input)
            dist = tfd.MultivariateNormalDiag(Input, sigma,
                                              validate_args=True,
                                              allow_nan_stats=False)
            dist = self.transformation.transform(dist)
            return dist

    def get_mvn_from_transformation(self, Input):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            sigma = None

            mu = self.transformation.transform(Input)
            if isinstance(mu, tuple):
                assert len(mu) == 2, "output of {} should contain 2 elements".format(self.transformation.name)
                mu, sigma = mu

            sigma_con = self.get_sigma(mu)

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
                    mvn = tfd.MultivariateNormalFullCovariance(mu, sigma,
                                                               validate_args=True,
                                                               allow_nan_stats=False)

            return mvn

    def get_sigma(self, mu):
        Dout = mu.shape.as_list()[-1]
        sigma_con = tf.get_variable("sigma_con",
                                    shape=[Dout],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(self.sigma_init),
                                    trainable=True)
        sigma_con = tf.nn.softplus(sigma_con)
        sigma_con = tf.where(tf.is_nan(sigma_con), tf.zeros_like(sigma_con), sigma_con)
        sigma_con = tf.maximum(sigma_con, self.sigma_min)
        return sigma_con

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
            if isinstance(self.transformation, NF):
                # for flow, choose the point with max prob
                sample = mvn.sample(self.transformation.sample_num)
                log_prob = mvn.log_prob(sample)
                ML_idx = tf.argmax(log_prob, axis=0, output_type=tf.int32)
                batch_shape = mvn.batch_shape
                meshgrid_axis = [tf.range(batch_axis_size) for batch_axis_size in batch_shape]
                gather_idx = tf.stack([ML_idx] + tf.meshgrid(*meshgrid_axis, indexing="ij"), axis=-1)
                return tf.gather_nd(sample, gather_idx)
            else:
                return mvn.mean()

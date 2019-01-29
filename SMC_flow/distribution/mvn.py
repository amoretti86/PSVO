import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer

from distribution.base import distribution
from flow import NF


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

    def get_dist(self, Input=None):
        if Input is None:
            assert self.output_0 is not None, "output_0 is not initialized for {}!".format(self.name)
        if isinstance(self.transformation, NF):
            dist = self.get_dist_from_flow(Input)
        else:
            dist = self.get_mvn_from_transformation(Input)
        return dist

    def get_dist_from_flow(self, Input=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if Input is None:
                mu = self.output_0
            else:
                mu = Input

            if mu.shape.as_list()[-1] != self.transformation.event_size:
                mu = fully_connected(mu, self.transformation.event_size,
                                     weights_initializer=xavier_initializer(uniform=False),
                                     biases_initializer=tf.constant_initializer(0),
                                     activation_fn=None,
                                     reuse=tf.AUTO_REUSE, scope="output")

            sigma = self.get_sigma(mu, sigma=None)

            dist = tfd.MultivariateNormalDiag(mu, sigma,
                                              validate_args=True,
                                              allow_nan_stats=False)
            if Input is not None:
                dist = self.transformation.transform(dist)
            return dist

    def get_mvn_from_transformation(self, Input=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            sigma = None

            if Input is None:
                mu = self.output_0
            else:
                mu = self.transformation.transform(Input)
                if isinstance(mu, tuple):
                    assert len(mu) == 2, "output of {} should contain 2 elements".format(self.transformation.name)
                    mu, sigma = mu

            sigma = self.get_sigma(mu, sigma)

            mvn = tfd.MultivariateNormalDiag(mu, sigma,
                                             validate_args=True,
                                             allow_nan_stats=False)
            return mvn

    def get_sigma(self, mu, sigma=None):
        if sigma is None:
            Dout = mu.shape.as_list()[-1]
            sigma = tf.get_variable("sigma",
                                    shape=[Dout],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(self.sigma_init),
                                    trainable=True)
            sigma = tf.nn.softplus(sigma)

        sigma = tf.where(tf.is_nan(sigma), tf.zeros_like(sigma), sigma)
        sigma = tf.maximum(sigma, self.sigma_min)

        return sigma

    def sample_and_log_prob(self, Input, sample_shape=(), name=None):
        dist = self.get_dist(Input)
        with tf.variable_scope(name or self.name):
            sample = dist.sample(sample_shape)
            log_prob = dist.log_prob(sample)
            return sample, log_prob

    def log_prob(self, Input, output, name=None):
        dist = self.get_dist(Input)
        with tf.variable_scope(name or self.name):
            return dist.log_prob(output)

    def mean(self, Input, name=None):
        dist = self.get_dist(Input)
        with tf.variable_scope(name or self.name):
            if isinstance(self.transformation, NF):
                # for flow, choose the point with max prob
                sample = dist.sample(self.transformation.sample_num)
                log_prob = dist.log_prob(sample)
                ML_idx = tf.argmax(log_prob, axis=0, output_type=tf.int32)
                batch_shape = dist.batch_shape
                meshgrid_axis = [tf.range(batch_axis_size) for batch_axis_size in batch_shape]
                gather_idx = tf.stack([ML_idx] + tf.meshgrid(*meshgrid_axis, indexing="ij"), axis=-1)
                return tf.gather_nd(sample, gather_idx)
            else:
                return dist.mean()

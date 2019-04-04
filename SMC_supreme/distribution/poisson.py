import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd

from SMC_supreme.distribution.base import distribution


class poisson(distribution):
    # multivariate poisson distribution

    def __init__(self, transformation):
        self.transformation = transformation

    def sample(self, Input):
        assert isinstance(Input, np.ndarray), "Input for poisson must be np.ndarray, {} is given".format(type(Input))

        def safe_softplus(x, limit=30):
            x[x < limit] = np.log(1.0 + np.exp(x[x < limit]))
            return x

        lambdas = safe_softplus(self.transformation.transform(Input))
        return np.random.poisson(lambdas)


class tf_poisson(distribution):
    # multivariate poisson distribution, can only be used as emission distribution

    def __init__(self, transformation, name='tf_poisson'):
        self.transformation = transformation
        self.name = name

    def get_poisson(self, Input):
        with tf.variable_scope(self.name):
            lambdas, _ = self.transformation.transform(Input)
            lambdas = tf.nn.softplus(lambdas) + 1e-6
            poisson = tfd.MultivariateNormalDiag(lambdas,
                                                 validate_args=True,
                                                 allow_nan_stats=False)
            return poisson

    def log_prob(self, Input, output, name=None):
        poisson = self.get_poisson(Input)
        with tf.variable_scope(name or self.name):
            return poisson.log_prob(output)

    def mean(self, Input, name=None):
        poisson = self.get_poisson(Input)
        with tf.variable_scope(name or self.name):
            return poisson.mean()

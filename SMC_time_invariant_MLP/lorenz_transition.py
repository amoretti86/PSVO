import numpy as np

import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.integrate import odeint as tf_odeint

class tf_lorenz:
    """
    Define multivariate normal density based on lorenz model
    """

    def __init__(self, n_particles, batch_size, lorenz_params, CovMat, output_0, name='tf_fhn', dtype=tf.float32):
        # A: tensor, shape = (Dout, Din)
        # Sigma: tensor, shape = (Dout, Dout)
        # output_0: tensor, shape = (batch_size, Dout)
        with tf.name_scope(name):
            self.n_particles = n_particles
            self.batch_size = batch_size
            (self.sigma, self.rho, self.beta, self.dt) = lorenz_params
            self.CovMat = CovMat
            self.output_0 = output_0
            self.Dout = self.Din = 3
            self.name = name
            self.dtype = dtype

    def get_mvn(self, Input, name):
        with tf.name_scope(self.name):
            if Input is None:
                mvn = tfd.MultivariateNormalFullCovariance(loc=self.output_0,
                                                           covariance_matrix=self.CovMat,
                                                           name="fhn_mvn")
            else:
                def lorenz_equation(state, t):
                    x, y, z = tf.unstack(state, axis = -1)
                    dx = self.sigma * (y - x)
                    dy = x * (self.rho - z) - y
                    dz = x * y - self.beta * z
                    return tf.stack([dx, dy, dz], axis = -1)

                t = np.arange(0.0, self.dt, self.dt)
                loc = tf_odeint(lorenz_equation, Input, t, name='loc')[0]
                mvn = tfd.MultivariateNormalFullCovariance(loc=loc,
                                                           covariance_matrix=self.CovMat,
                                                           name="fhn_mvn")
            return mvn

    def sample(self, Input, name=None):
        # Input:  tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
        # sample: tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
        mvn = self.get_mvn(Input, name)
        with tf.name_scope(name or self.name):
            if Input is None:
                return mvn.sample((self.n_particles), name="samples")
            else:
                return mvn.sample(name="samples")

    def prob(self, Input, output, name=None):
        # Input: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
        # output: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
        # prob:		tensor, shape = (n_particles, batch_size), dtype = self.dtype
        mvn = self.get_mvn(Input, name)
        with tf.name_scope(name or self.name):
            if Input is None:
                return mvn.prob(output, name="prob")
            else:
                return mvn.prob(output, name="prob")

    def log_prob(self, Input, output, name=None):
        # Input: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
        # output: 	tensor, shape = (n_particles, batch_size, Dx), dtype = self.dtype
        # prob:		tensor, shape = (n_particles, batch_size), dtype = self.dtype
        mvn = self.get_mvn(Input, name)
        with tf.name_scope(name or self.name):
            if Input is None:
                return mvn.log_prob(output, name="log_prob")
            else:
                return mvn.log_prob(output, name="log_prob")
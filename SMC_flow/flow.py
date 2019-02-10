import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb


class BatchNorm(tfb.Bijector):
    def __init__(self, eps=1e-5, decay=0.95, validate_args=False, name="batch_norm"):
        super(BatchNorm, self).__init__(
            event_ndims=1, validate_args=validate_args, name=name)
        self._vars_created = False
        self.eps = eps
        self.decay = decay

    def _create_vars(self, x):
        n = x.get_shape().as_list()[1]
        with tf.variable_scope(self.name):
            self.beta = tf.get_variable('beta', [1, n], dtype=tf.float32)
            self.gamma = tf.get_variable('gamma', [1, n], dtype=tf.float32)
            self.train_m = tf.get_variable(
                'mean', [1, n], dtype=tf.float32, trainable=False)
            self.train_v = tf.get_variable(
                'var', [1, n], dtype=tf.float32, initializer=tf.ones_initializer, trainable=False)
        self._vars_created = True

    def _forward(self, u):
        if not self._vars_created:
            self._create_vars(u)
        return (u - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.train_v + self.eps) + self.train_m

    def _inverse(self, x):
        # Eq 22. Called during training of a normalizing flow.
        if not self._vars_created:
            self._create_vars(x)
        # statistics of current minibatch
        m, v = tf.nn.moments(x, axes=[0], keep_dims=True)
        # update train statistics via exponential moving average
        update_train_m = tf.assign_sub(
            self.train_m, self.decay * (self.train_m - m))
        update_train_v = tf.assign_sub(
            self.train_v, self.decay * (self.train_v - v))
        # normalize using current minibatch statistics, followed by BN scale and shift
        with tf.control_dependencies([update_train_m, update_train_v]):
            return (x - m) * 1. / tf.sqrt(v + self.eps) * tf.exp(self.gamma) + self.beta

    def _inverse_log_det_jacobian(self, x):
        # at training time, the log_det_jacobian is computed from statistics of the
        # current minibatch.
        if not self._vars_created:
            self._create_vars(x)
        _, v = tf.nn.moments(x, axes=[0], keep_dims=True)
        abs_log_det_J_inv = tf.reduce_sum(
            self.gamma - .5 * tf.log(v + self.eps))
        return abs_log_det_J_inv


class NF:
    def __init__(self,
                 n_layers,
                 event_size,
                 hidden_layers=[512, 512],
                 sample_num=100,
                 flow_type="IAF",
                 name="NF",
                 use_batchnorm=True):
        self.event_size = event_size
        self.sample_num = sample_num
        self.flow_type = flow_type
        self.name = name
        self.bijector = self.init_bijectors(n_layers, hidden_layers)
        self.use_batchnorm = use_batchnorm

    @staticmethod
    def init_once(x, name):
        return tf.get_variable(name, dtype=tf.int32, initializer=x, trainable=False)

    def init_bijectors(self, n_layers, hidden_layers):
        with tf.variable_scope(self.name):
            bijectors = []
            for i in range(n_layers):
                if self.flow_type == "MAF":
                    bijectors.append(tfb.MaskedAutoregressiveFlow(
                        shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                            hidden_layers=hidden_layers,
                            activation=tf.nn.leaky_relu,
                            name="MAF_template_{}".format(i)),
                        name="MAF_{}".format(i)))
                elif self.flow_type == "IAF":
                    bijectors.append(
                        tfb.Invert(
                            tfb.RealNVP(
                                shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                                    hidden_layers=hidden_layers,
                                    activation=tf.nn.leaky_relu,
                                    name="MAF_template_{}".format(i))
                            ),
                            name="IAF_{}".format(i)
                        )
                    )
                elif self.flow_type == "RealNVP":
                    bijectors.append(
                        tfb.Invert(
                            tfb.MaskedAutoregressiveFlow(
                                shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                                    hidden_layers=hidden_layers,
                                    activation=tf.nn.leaky_relu,
                                    name="MAF_template_{}".format(i))
                            ),
                            name="RealNVP_{}".format(i)
                        )
                    )
                if self.use_batchnorm and i % 2 == 0:
                    bijectors.append(BatchNorm(name="BatchNorm_{}".format(i)))
                bijectors.append(tfb.Permute(permutation=list(range(1, self.event_size)) + [0]))

            flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])))

            return flow_bijector

    def transform(self, base_dist, name=None):
        dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=self.bijector,
            name=name)

        return dist

if __name__ == "__main__":
    Dx = 2
    batch_size = 5
    n_particles = 10
    flow = NF(2, Dx)
    mvn = tfd.MultivariateNormalDiag(tf.zeros((n_particles, batch_size, Dx)), name="mvn")
    dist, sample, log_prob = flow.sample_and_log_prob(mvn, name="trans_mvn")
    print(dist)
    print(sample)
    print(log_prob)

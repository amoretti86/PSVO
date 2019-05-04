import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb


class NF:
    def __init__(self,
                 n_layers,
                 event_size,
                 flow_to_reverse=None,
                 hidden_layers=[32],
                 sample_num=100,
                 flow_type="IAF",
                 shift_only=False,
                 log_scale_min_clip=-0.1,
                 log_scale_max_clip=0.1,
                 log_scale_clip_gradient=False,
                 name="NF"):
        if flow_to_reverse is None:
            self.event_size              = event_size
            self.sample_num              = sample_num
            self.flow_type               = flow_type
            self.shift_only              = shift_only
            self.log_scale_min_clip      = log_scale_min_clip
            self.log_scale_max_clip      = log_scale_max_clip
            self.log_scale_clip_gradient = log_scale_clip_gradient
            self.name                    = name
            self.bijector                = self.init_bijectors(n_layers, hidden_layers)
        else:
            self.event_size = flow_to_reverse.event_size
            self.sample_num = flow_to_reverse.sample_num
            self.flow_type  = flow_to_reverse.flow_type + "_reversed"
            self.name       = flow_to_reverse.name + "_reversed"
            self.bijector   = tfb.Invert(flow_to_reverse.bijector)

    @staticmethod
    def init_once(x, name):
        return tf.get_variable(name, dtype=tf.int32, initializer=x, trainable=False)

    def init_bijectors(self, n_layers, hidden_layers):
        with tf.variable_scope(self.name):
            bijectors = []
            for i in range(n_layers):
                if self.flow_type == "MAF":
                    bijectors.append(
                        tfb.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                                hidden_layers=hidden_layers,
                                activation=tf.nn.relu,
                                log_scale_min_clip=self.log_scale_min_clip,
                                log_scale_max_clip=self.log_scale_max_clip,
                                shift_only=self.shift_only,
                                log_scale_clip_gradient=self.log_scale_clip_gradient,
                                name="MAF_template_{}".format(i)
                            ),
                            name="MAF_{}".format(i)
                        )
                    )
                elif self.flow_type == "IAF":
                    bijectors.append(
                        tfb.Invert(
                            tfb.MaskedAutoregressiveFlow(
                                shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                                    hidden_layers=hidden_layers,
                                    activation=tf.nn.relu,
                                    log_scale_min_clip=self.log_scale_min_clip,
                                    log_scale_max_clip=self.log_scale_max_clip,
                                    shift_only=self.shift_only,
                                    log_scale_clip_gradient=self.log_scale_clip_gradient,
                                    name="MAF_template_{}".format(i))
                            ),
                            name="IAF_{}".format(i)
                        )
                    )
                elif self.flow_type == "RealNVP":
                    bijectors.append(
                        tfb.RealNVP(
                            num_masked=self.event_size - 1,
                            shift_and_log_scale_fn=tfb.real_nvp_default_template(
                                hidden_layers=hidden_layers,
                                activation=tf.nn.relu,
                                shift_only=self.shift_only,
                                name="RealNVP_template_{}".format(i)
                            ),
                            name="RealNVP_{}".format(i)
                        )
                    )
                else:
                    raise ValueError("Unknown flow type {}".format(self.flow_type))
                bijectors.append(tfb.Permute(permutation=list(range(1, self.event_size)) + [0]))
                # bijectors.append(
                #     tfb.Permute(
                #         self.init_once(np.random.permutation(self.event_size).astype("int32"),
                #                        name="permutation_{}".format(i))
                #     )
                # )

            flow_bijector = tfb.Chain(list(reversed(bijectors[:-1])), validate_args=True, name="NF_chain")

            return flow_bijector

    def transform(self, base_dist, name=None):
        dist = tfd.TransformedDistribution(
            distribution=base_dist,
            bijector=self.bijector,
            validate_args=True,
            name=name or self.name)

        return dist

if __name__ == "__main__":
    Dx = 2
    batch_size = 1
    n_particles = 2
    flow = NF(4, Dx)

    mvn = tfd.MultivariateNormalDiag(tf.zeros((n_particles, batch_size, Dx)), name="mvn")
    dist = flow.transform(mvn, name="trans_mvn")
    x = dist.sample(5)
    log_prob = dist.log_prob(x)

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    sess.run(init)

    print(dist)
    print(x.eval())
    print(log_prob.eval())

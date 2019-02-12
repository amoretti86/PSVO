"""Testing norm_flow.py classes."""


import numpy as np
import tensorflow as tf

from norm_flow import MultiLayerPlanarFlow

def test_planar_flow():
    """Tester function for planar flow on tensor."""

    diag_norm = tf.contrib.distributions.MultivariateNormalDiag

    # number of examples to be transformed.
    n_sample = 10000
    # number of parallel normalizing flows to be used.
    n_flow = 3
    # dimension of the space.
    dim = 2
    # number of layers per normalizing flow.
    n_layer = 4

    with tf.Graph().as_default():
        dist = diag_norm(loc=np.ones(dim), scale_diag=np.ones(dim))
        # input noise to the normalizing flow.
        x = dist.sample([n_flow, n_sample])
        p = dist.log_prob(x)

        mpf = MultiLayerPlanarFlow(
            dim=dim, num_layer=n_layer, n_flow=n_flow,
            non_linearity=tf.nn.softplus)
        y = mpf.operator(x)
        pp = mpf.log_det_jacobian(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x_, p_, y_, pp_ = sess.run([x, p, y, pp])

    # check the shape of output of transfomration.
    assert(y_.shape == (n_flow, n_sample, dim))
    # check the shape of output log-det-jacobian
    assert(pp_.shape == (n_flow, n_sample))

if __name__ == "__main__":
    test_planar_flow()

import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd


# test resampling from X_T

def test_resampling_particles():
    n_particles = 3
    batch_size = 1
    Dx = 2
    M = 2

    X_t = np.reshape(np.arange(1,7), (n_particles, batch_size, Dx))
    """
    X_t = [ [[1,2]], [[3,4]], [[5,6]] ]
    """

    X_t = tf.constant(X_t, name="X_t")

    idx = np.array([[1], [0]])  # (M, batch_size)
    """
    idx = [ [1], [0] ]
    """

    resampled_rslt = tf.gather_nd(X_t, idx)

    with tf.Session() as sess:
        rslt = sess.run(resampled_rslt)

    assert rslt.shape == (M, batch_size, Dx)

    np.testing.assert_array_equal(rslt, np.array([[[3, 4]], [[1, 2]]]))


def test_sample_weights():
    M = 2
    n_particles = 3
    batch_size = 1
    weights = np.reshape(np.arange(0, 6), (M, n_particles, batch_size))
    """
    weights = [[[0], [1], [2]],
                [[3], [4], [5]]]
    """
    idx = np.array([[2], [0]])  # shape (M, batch_size)

    idx_reformat = [(i, idx[i][j], j) for i in range(M) for j in range(batch_size)]

    rslt = tf.gather_nd(weights, idx_reformat)
    rslt = tf.reshape(rslt, (M, batch_size))

    with tf.Session() as sess:
        output = sess.run(rslt)

    np.testing.assert_array_equal(output, [[2], [3]])


test_sample_weights()
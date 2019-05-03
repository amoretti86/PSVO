import tensorflow as tf
import numpy as np
from tensorflow_probability import distributions as tfd


# test resampling from X_T

def test_resampling_particles():
    """
    Gather M by batch_size particles, of shape (M, batch_size, Dx),
    from K by batch_size particles, of shape (n_particles, batch_size, Dx),
    according to sample_idx of shape (M, batch_size, Dx)
    :return:
    """
    n_particles = 3
    batch_size = 2
    Dx = 4
    M = 3

    X_t = np.reshape(np.arange(0, n_particles*batch_size*Dx), (n_particles, batch_size, Dx))

    """
    X_t = [[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]],

           [[ 8,  9, 10, 11],
           [12, 13, 14, 15]],

           [[16, 17, 18, 19],
           [20, 21, 22, 23]]]
    """

    X_t = tf.constant(X_t, name="X_t")

    idx = np.array([[1, 0], [2, 1], [0, 1]])  # (M, batch_size)

    """
    X_t[k][batch]: shape (4,)
    m = 0, batch = 0, k = 1 --> [8, 9, 10, 11]
    m = 0, batch = 1, k = 0 --> [4, 5, 6, 7]
    
    m = 1, batch = 0, k = 2 --> [16, 17, 18, 19]
    m = 1, batch = 1, k = 1 --> [12, 13, 14, 15]
    
    m = 2, batch = 0, k = 0 --> [0, 1, 2, 3]
    m = 2, batch = 1, k = 1 --> [12, 13, 14, 15]
    """

    correct_output = np.array([[[ 8, 9, 10, 11],
                                [ 4, 5, 6, 7]],
                               [[16, 17, 18, 19],
                                [12, 13, 14, 15]],
                               [[0,  1,  2,  3],
                                [12, 13, 14, 15]]])
    # shape (M, batch_size, Dx)

    idx_reformat = [[[idx[i][j], j] for j in range(batch_size)] for i in range(M)]

    resampled_rslt = tf.gather_nd(X_t, idx_reformat)

    with tf.Session() as sess:
        rslt = sess.run(resampled_rslt)

    assert rslt.shape == (M, batch_size, Dx)

    np.testing.assert_array_equal(rslt, correct_output)


def test_sample_weights():
    """
    Gather M by batch_size weights from M by K by batch_size weights,
    according to sample_index of shape (M, batch_size)
    """

    M = 3
    n_particles = 4
    batch_size = 2
    weights = np.reshape(np.arange(0, M*n_particles*batch_size), (M, n_particles, batch_size))
    """
    weights = [[[ 0,  1],
        [ 2,  3],
        [ 4,  5],
        [ 6,  7]],

       [[ 8,  9],
        [10, 11],
        [12, 13],
        [14, 15]],

       [[16, 17],
        [18, 19],
        [20, 21],
        [22, 23]]]
    """
    idx = np.array([[2, 0], [0, 1], [3, 2]])  # shape (M, batch_size)

    """
    expected output shape (reduces dimension of n_particles): (M, batch_size)
    m=0, batch=0, k=2 --> 4
    m=0, batch=1, k=0 --> 1
    m=1, batch=0, k=0 --> 8
    m=1, batch=1, k=1 --> 11
    m=2, batch=0, k=3 --> 22
    m=2, batch=1, k=2 --> 21
    
    """
    correct_output = np.array([[4, 1], [8, 11], [22, 21]])

    idx_reformat = [[(i, idx[i][j], j) for j in range(batch_size)] for i in range(M)]
    """
    [[(0, 2, 0), (0, 0, 1)],
     [(1, 0, 0), (1, 1, 1)],
      [(2, 3, 0), (2, 2, 1)]]
    """

    rslt = tf.gather_nd(weights, idx_reformat)
    rslt = tf.reshape(rslt, (M, batch_size))

    with tf.Session() as sess:
        output = sess.run(rslt)

    np.testing.assert_array_equal(output, correct_output)


test_resampling_particles()
test_sample_weights()

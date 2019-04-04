import numpy as np
import tensorflow as tf

from SMC_supreme.transformation.base import transformation


class linear_transformation(transformation):

    def transform(self, Input):
        '''
        Integrates the Lorenz ODEs
        '''
        A = self.params
        return np.dot(A, Input)


class tf_linear_transformation(transformation):

    def transform(self, Input):
        '''
        Integrates the Lorenz ODEs
        '''
        A = self.params
        Input_shape = Input.shape.as_list()
        output_shape = list(Input_shape)
        output_shape[-1] = A.shape.as_list()[0]

        Input_reshaped = tf.reshape(Input, [-1, Input_shape[-1]])
        output_reshaped = tf.matmul(Input_reshaped, A, transpose_b=True)
        output = tf.reshape(output_reshaped, output_shape)
        return output


# test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    A = np.array([[0.99, 0.01], [0.01, 0.99]])
    Dx = 2
    T = 100
    batch_size = 10

    # for np ver
    linear = linear_transformation(A)

    X = np.zeros((T, Dx))
    X[0] = np.random.uniform(low=0, high=1, size=Dx)
    for t in range(1, T):
        X[t] = linear.transform(X[t - 1])

    plt.figure()
    plt.plot(X[:, 0], X[:, 1])
    plt.show()

    # for tf ver
    A = tf.constant(A, dtype=tf.float32)
    tf_linear = tf_linear_transformation(A)

    Xs = []
    X = tf.constant(np.random.uniform(low=-1, high=1, size=(batch_size, Dx)), dtype=tf.float32)
    for t in range(1, T):
        X = tf_linear.transform(X)
        Xs.append(X)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    Xs = tf.stack(Xs, axis=1).eval()

    plt.figure()
    for i in range(batch_size):
        plt.plot(Xs[i, :, 0], Xs[i, :, 1])
    plt.show()

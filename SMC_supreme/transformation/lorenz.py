import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
from tensorflow.contrib.integrate import odeint as tf_odeint

from SMC_supreme.transformation.base import transformation


class lorenz_transformation(transformation):

    def transform(self, X_prev):
        '''
        Integrates the lorenz ODEs
        '''
        sigma, rho, beta, dt = self.params

        def lorenz_equation(X, t, sigma, rho, beta):
            x, y, z = X

            xd = sigma * (y - x)
            yd = (rho - z) * x - y
            zd = x * y - beta * z

            return [xd, yd, zd]

        t = np.arange(0, 2 * dt, dt)
        X = odeint(lorenz_equation, X_prev, t, args=(sigma, rho, beta))[1, :]

        return X


class tf_lorenz_transformation(transformation):

    def transform(self, X_prev):
        """
        X_prev.shape = [B0, B1, ..., Bn, Dx]
        """
        sigma, rho, beta, dt = self.params

        def lorenz_equation(X, t):
            x, y, z = tf.unstack(X, axis=-1)

            xd = sigma * (y - x)
            yd = (rho - z) * x - y
            zd = x * y - beta * z

            return tf.stack([xd, yd, zd], axis=-1)

        t = np.arange(0.0, 2 * dt, dt)
        X = tf.unstack(tf_odeint(lorenz_equation, X_prev, t, name="loc"), axis=0)[1]

        return X


# test code
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tensorflow as tf
    lorenz_params = (10.0, 28.0, 8.0 / 3.0, 0.01)
    Dx = 3
    T = 1500
    batch_size = 10

    # for np ver
    lorenz = lorenz_transformation(lorenz_params)

    X = np.zeros((T, Dx))
    X[0] = np.random.uniform(low=0, high=1, size=Dx)
    for t in range(1, T):
        X[t] = lorenz.transform(X[t - 1])

    # plt.figure()
    # plt.plot(X[:, 0], X[:, 1])
    # plt.show()

    # for tf ver
    tf_lorenz = tf_lorenz_transformation(lorenz_params)

    Xs = []
    X = tf.constant(np.random.uniform(low=-1, high=1, size=(batch_size, Dx)), dtype=tf.float32)
    for t in range(1, T):
        X = tf_lorenz.transform(X)
        Xs.append(X)

    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    Xs = tf.stack(Xs, axis=1).eval()

    plt.figure()
    for i in range(batch_size):
        plt.plot(Xs[i, :, 0], Xs[i, :, 1])
    plt.show()

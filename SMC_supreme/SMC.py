import tensorflow as tf
from tensorflow_probability import distributions as tfd


class SMC:
    def __init__(self, q, f, g,
                 n_particles, batch_size,
                 encoder_cell=None,
                 q_takes_y=True,
                 q_use_true_X=False,
                 use_stop_gradient=False,
                 name="log_ZSMC"):
        self.q = q
        self.f = f
        self.g = g
        self.n_particles = n_particles
        self.batch_size = batch_size

        self.encoder_cell = encoder_cell

        self.q_takes_y = q_takes_y
        self.q_use_true_X = q_use_true_X
        self.use_stop_gradient = use_stop_gradient

        self.name = name

    def get_log_ZSMC(self, obs, x_0, hidden, q_cov=1):
        """
        Input:
            obs.shape = (batch_size, time, Dy)
        Output:
            log_ZSMC: shape = scalar
            log: stuff to debug
        """
        with tf.variable_scope(self.name):
            batch_size, time, Dy = obs.get_shape().as_list()
            batch_size, Dx = x_0.get_shape().as_list()

            Xs = []
            log_Ws = []
            Ws = []
            fs = []
            gs = []
            qs = []

            if self.encoder_cell is not None:
                self.encoder_cell.encode(obs[:, 0:-1], x_0)

            log_ZSMC = 0
            X_prev = None

            for t in range(0, time):
                # when t = 1, sample with x_0
                # otherwise, sample with X_prev
                if t == 0:
                    sample_size = (self.n_particles)
                else:
                    sample_size = ()

                if self.q_takes_y:
                    if t == 0:
                        q_t_Input = tf.concat([x_0, obs[:, 0]], axis=-1)
                    else:
                        y_t_expanded = tf.tile(tf.expand_dims(obs[:, t], axis=0), (self.n_particles, 1, 1))
                        q_t_Input = tf.concat([X_prev, y_t_expanded], axis=-1)
                else:
                    q_t_Input = X_prev

                if self.q_use_true_X:
                    mvn = tfd.MultivariateNormalFullCovariance(hidden[:, t, :], q_cov * tf.eye(Dx),
                                                               name="q_{}_mvn".format(t))
                    X = mvn.sample((self.n_particles))
                    q_t_log_prob = mvn.log_prob(X)
                else:
                    X, q_t_log_prob = self.q.sample_and_log_prob(q_t_Input, sample_shape=sample_size,
                                                                 name="q_{}_sample_and_log_prob".format(t))

                f_t_log_prob = self.f.log_prob(X_prev, X, name="f_{}_log_prob".format(t))
                g_t_log_prob = self.g.log_prob(X, obs[:, t], name="g_{}_log_prob".format(t))

                log_W = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_W_{}".format(t))
                W = tf.exp(log_W, name="W_{}".format(t))
                log_ZSMC += tf.log(tf.reduce_mean(W, axis=0, name="W_{}_mean".format(t)),
                                   name="log_ZSMC_{}".format(t))

                qs.append(q_t_log_prob)
                fs.append(f_t_log_prob)
                gs.append(g_t_log_prob)
                log_Ws.append(log_W)
                Ws.append(W)

                # no need to resample for t = time - 1
                if t == time - 1:
                    break

                log_W = tf.transpose(log_W)
                categorical = tfd.Categorical(logits=log_W, validate_args=True,
                                              name="Categorical_{}".format(t))
                if self.use_stop_gradient:
                    idx = tf.stop_gradient(categorical.sample(self.n_particles))  # (n_particles, batch_size)
                else:
                    idx = categorical.sample(self.n_particles)

                # ugly stuff used to resample X
                batch_1xB = tf.expand_dims(tf.range(batch_size), axis=0)       # (1, batch_size)
                batch_NxB = tf.tile(batch_1xB, (self.n_particles, 1))          # (n_particles, batch_size)

                idx_NxBx1 = tf.expand_dims(idx, axis=2)                        # (n_particles, batch_size, 1)
                batch_NxBx1 = tf.expand_dims(batch_NxB, axis=2)                # (n_particles, batch_size, 1)

                final_idx_NxBx2 = tf.concat((idx_NxBx1, batch_NxBx1), axis=2)  # (n_particles, batch_size, 2)
                X_prev = tf.gather_nd(X, final_idx_NxBx2)                      # (n_particles, batch_size, Dx)

                # collect X after rather than before resampling
                Xs.append(X_prev)

            # to make sure len(Xs) = time
            Xs.append(X)

            qs = tf.stack(qs)
            fs = tf.stack(fs)
            gs = tf.stack(gs)
            log_Ws = tf.stack(log_Ws)
            Ws = tf.stack(Ws)
            Xs = tf.stack(Xs)

            qs = tf.transpose(qs, perm=[2, 0, 1], name="qs")                # (batch_size, time, n_particles)
            fs = tf.transpose(fs, perm=[2, 0, 1], name="fs")                # (batch_size, time, n_particles)
            gs = tf.transpose(gs, perm=[2, 0, 1], name="gs")                # (batch_size, time, n_particles)
            log_Ws = tf.transpose(log_Ws, perm=[2, 0, 1], name="log_Ws")    # (batch_size, time, n_particles)
            Ws = tf.transpose(Ws, perm=[2, 0, 1], name="Ws")                # (batch_size, time, n_particles)
            Xs = tf.transpose(Xs, perm=[2, 0, 1, 3], name="Xs")             # (batch_size, time, n_particles, Dx)

            mean_log_ZSMC = tf.reduce_mean(log_ZSMC, name="mean_log_ZSMC")

        return mean_log_ZSMC, [Xs, log_Ws, Ws, fs, gs, qs]

    def n_step_MSE(self, n_steps, hidden, obs):
        # Compute MSE_k for k = 0, ..., n_steps
        batch_size, time, Dx = hidden.shape.as_list()
        batch_size, time, Dy = obs.shape.as_list()
        assert n_steps < time, "n_steps = {} >= time".format(n_steps)
        with tf.variable_scope(self.name):

            x_BxTmkxDz = hidden

            # get y_hat
            y_hat_N_BxTxDy = []

            for k in range(n_steps):
                y_hat_BxTmkxDy = self.g.mean(x_BxTmkxDz)                            # (batch_size, time - k, Dy)
                y_hat_N_BxTxDy.append(y_hat_BxTmkxDy)

                x_Tmk_BxDz = tf.unstack(x_BxTmkxDz, axis=1, name="x_Tmk_BxDz")      # list of (batch_size, Dx)
                x_BxTmkxDz = tf.stack(x_Tmk_BxDz[:-1], axis=1, name="x_BxTmkxDz")   # (batch_size, time - k - 1, Dx)
                x_BxTmkxDz = self.f.mean(x_BxTmkxDz)                                # (batch_size, time - k - 1, Dx)

            y_hat_BxTmNxDy = self.g.mean(x_BxTmkxDz)                                # (batch_size, time - N, Dy)
            y_hat_N_BxTxDy.append(y_hat_BxTmNxDy)

            # get y_true
            y_N_BxTxDy = []
            for k in range(n_steps + 1):
                y_BxTmkxDy = obs[:, k:, :]
                y_N_BxTxDy.append(y_BxTmkxDy)

            # compare y_hat and y_true to get MSE_k, y_mean, y_var
            # FOR THE BATCH and FOR k = 0, ..., n_steps

            MSE_ks = []     # [MSE_0, MSE_1, ..., MSE_N]
            y_means = []    # [y_mean_0 (shape = Dy), ..., y_mean_N], used to calculate y_var across all batches
            y_vars = []     # [y_var_0 (shape = Dy), ..., y_var_N], used to calculate y_var across all batches
            for k, (y_hat_BxTmkxDy, y_BxTmkxDy) in enumerate(zip(y_hat_N_BxTxDy, y_N_BxTxDy)):
                MSE_k = tf.reduce_sum((y_hat_BxTmkxDy - y_BxTmkxDy)**2, name="MSE_{}".format(k))
                MSE_ks.append(MSE_k)
                y_mean = tf.reduce_mean(y_BxTmkxDy, axis=[0, 1], name="y_mean_{}".format(k))
                y_means.append(y_mean)
                y_var = tf.reduce_sum((y_BxTmkxDy - y_mean)**2, axis=[0, 1], name="y_var_{}".format(k))
                y_vars.append(y_var)

            MSE_ks = tf.stack(MSE_ks, name="MSE_ks")     # (n_steps + 1)
            y_means = tf.stack(y_means, name="y_means")  # (n_steps + 1, Dy)
            y_vars = tf.stack(y_vars, name="y_vars")     # (n_steps + 1, Dy)

            return MSE_ks, y_means, y_vars, y_hat_N_BxTxDy

    def get_nextX(self, X):
        with tf.variable_scope(self.name):
            return self.f.mean(X)

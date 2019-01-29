import tensorflow as tf
from tensorflow_probability import distributions as tfd


class SMC:
    def __init__(self, q, f, g,
                 n_particles,
                 q2=None,
                 smoothing=False,
                 q_takes_y=True,
                 q_uses_true_X=False,
                 use_stop_gradient=False,
                 smoothing_perc=1.0,
                 name="log_ZSMC"):

        self.q = q
        self.f = f
        self.g = g
        self.q2 = q2
        self.n_particles = n_particles

        self.smoothing = smoothing
        self.q_takes_y = q_takes_y
        self.q_uses_true_X = q_uses_true_X
        self.use_stop_gradient = use_stop_gradient

        self.smoothing_perc = smoothing_perc
        self.name = name

    def get_log_ZSMC(self, obs, hidden, q_cov=1):
        """
        Input:
            obs.shape = (batch_size, time, Dy)
        Output:
            log_ZSMC: shape = scalar
            log: stuff to debug
        """
        with tf.variable_scope(self.name):
            batch_size, time, Dy = obs.get_shape().as_list()
            batch_size, Dx = self.q.output_0.get_shape().as_list()
            n_particles = self.n_particles

            X_prevs = []
            X_ancestors = []
            log_Ws = []
            qs = []
            fs = []
            gs = []

            all_fs = []

            X_ancestor = None
            X_prev = None
            resample_idx = None

            for t in range(0, time):
                # when t = 1, sample with x_0
                # otherwise, sample with X_ancestor
                if t == 0:
                    sample_size = (n_particles)
                else:
                    sample_size = ()

                if self.q_takes_y:
                    if t == 0:
                        q_t_Input = tf.concat([self.q.output_0, obs[:, 0]], axis=-1)
                    else:
                        y_t_expanded = tf.tile(tf.expand_dims(obs[:, t], axis=0), (n_particles, 1, 1))
                        q_t_Input = tf.concat([X_ancestor, y_t_expanded], axis=-1)
                else:
                    q_t_Input = X_ancestor

                if self.q_uses_true_X:
                    mvn = tfd.MultivariateNormalFullCovariance(hidden[:, t, :], q_cov * tf.eye(Dx),
                                                               name="q_{}_mvn".format(t))
                    X = mvn.sample((n_particles))
                    q_t_log_prob = mvn.log_prob(X)
                else:
                    if self.q2 is None:
                        X, q_t_log_prob = self.q.sample_and_log_prob(q_t_Input, sample_shape=sample_size,
                                                                     name="q_{}_sample_and_log_prob".format(t))
                    else:
                        X, q_t_log_prob, f_t_log_prob = self.sample_from_2_q(X_ancestor, obs[:, t], sample_size)

                if self.smoothing and t != 0:
                    X_tile = tf.tile(tf.expand_dims(X, axis=1), (1, n_particles, 1, 1), name="X_tile")
                    f_t_all_log_prob = self.f.log_prob(X_prev, X_tile, name="f_{}_all_log_prob".format(t))
                    all_fs.append(f_t_all_log_prob)

                    f_t_log_prob = self.f.log_prob(X_ancestor, X, name="f_{}_log_prob".format(t))

                elif self.q2 is None:
                    f_t_log_prob = self.f.log_prob(X_ancestor, X, name="f_{}_log_prob".format(t))
                else:
                    # if q uses 2 networks, f_t_log_prob is already calculated above
                    pass

                g_t_log_prob = self.g.log_prob(X, obs[:, t], name="g_{}_log_prob".format(t))

                log_W = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_W_{}".format(t))

                log_Ws.append(log_W)
                qs.append(q_t_log_prob)
                fs.append(f_t_log_prob)
                gs.append(g_t_log_prob)

                # no need to resample for t = time - 1
                if t == time - 1:
                    break

                resample_idx = self.get_resample_idx(log_W, t)
                X_ancestor = tf.gather_nd(X, resample_idx)                    # (n_particles, batch_size, Dx)
                X_prev = X

                # collect X after rather than before resampling
                X_prevs.append(X_prev)
                X_ancestors.append(X_ancestor)

            # to make sure len(Xs) = time
            X_prevs.append(X_prev)
            X_ancestors.append(X_ancestor)

            if self.smoothing:
                reweighted_log_Ws = self.reweight_log_Ws(log_Ws, all_fs)
                log_ZSMC = self.compute_log_ZSMC(reweighted_log_Ws)

                Xs = []
                for t in range(time):
                    smoothing_idx = self.get_resample_idx(reweighted_log_Ws[t])
                    smoothed_X_t = tf.gather_nd(X_prevs[t], smoothing_idx)
                    Xs.append(smoothed_X_t)

                reweighted_log_Ws = tf.stack(reweighted_log_Ws)
                reweighted_log_Ws = tf.transpose(reweighted_log_Ws, perm=[2, 0, 1], name="reweighted_log_Ws")
                log = [reweighted_log_Ws]
            else:
                log_ZSMC = self.compute_log_ZSMC(log_Ws)
                Xs = X_ancestors
                log = []

            log_Ws = tf.stack(log_Ws)
            qs = tf.stack(qs)
            fs = tf.stack(fs)
            gs = tf.stack(gs)

            # (batch_size, time, n_particles, Dx)
            log_Ws = tf.transpose(log_Ws, perm=[2, 0, 1], name="log_Ws")    # (batch_size, time, n_particles)
            qs = tf.transpose(qs, perm=[2, 0, 1], name="qs")                # (batch_size, time, n_particles)
            fs = tf.transpose(fs, perm=[2, 0, 1], name="fs")                # (batch_size, time, n_particles)
            gs = tf.transpose(gs, perm=[2, 0, 1], name="gs")                # (batch_size, time, n_particles)

        return log_ZSMC, [Xs, X_ancestors, log_Ws, fs, gs, qs] + log

    def sample_from_2_q(self, X_ancestor, y_t, sample_size):
        q1_mvn = self.q.get_mvn(X_ancestor)
        q2_mvn = self.q2.get_mvn(y_t)

        assert isinstance(q1_mvn, tfd.MultivariateNormalDiag) and isinstance(q2_mvn, tfd.MultivariateNormalDiag)

        q1_mvn_mean, q1_mvn_cov = q1_mvn.mean(), q1_mvn.stddev()
        q2_mvn_mean, q2_mvn_cov = q2_mvn.mean(), q2_mvn.stddev()

        q1_mvn_cov_inv, q2_mvn_cov_inv = 1 / q1_mvn_cov, 1 / q2_mvn_cov
        combined_cov = 1 / (q1_mvn_cov_inv + q2_mvn_cov_inv)
        combined_mean = combined_cov * (q1_mvn_cov_inv * q1_mvn_mean + q2_mvn_cov_inv * q2_mvn_mean)

        mvn = tfd.MultivariateNormalDiag(combined_mean,
                                         combined_cov,
                                         validate_args=True,
                                         allow_nan_stats=False)
        X = mvn.sample(sample_size)
        q_t_log_prob = mvn.log_prob(X)
        f_t_log_prob = q1_mvn.log_prob(X)

        return X, q_t_log_prob, f_t_log_prob

    def get_resample_idx(self, log_W, t):
        n_particles, batch_size = log_W.get_shape().as_list()

        log_W_max = tf.stop_gradient(tf.reduce_max(log_W, axis=0), name="log_W_max")
        log_W = tf.transpose(log_W - log_W_max)
        categorical = tfd.Categorical(logits=log_W, validate_args=True,
                                      name="Categorical_{}".format(t))

        # sample multiple times to remove idx out of range
        idx = tf.ones((n_particles, batch_size), dtype=tf.int32) * n_particles
        for _ in range(3):
            if self.use_stop_gradient:
                fixup_idx = tf.stop_gradient(categorical.sample(n_particles))  # (n_particles, batch_size)
            else:
                fixup_idx = categorical.sample(n_particles)
            idx = tf.where(idx >= n_particles, fixup_idx, idx)

        # if still got idx out of range, replace them with idx from uniform distribution
        final_fixup = tf.random.uniform((n_particles, batch_size),
                                        maxval=n_particles, dtype=tf.int32)
        idx = tf.where(idx >= n_particles, final_fixup, idx)

        # ugly stuff used to resample X
        batch_1xB = tf.expand_dims(tf.range(batch_size), axis=0)            # (1, batch_size)
        batch_NxB = tf.tile(batch_1xB, (n_particles, 1))                    # (n_particles, batch_size)

        resample_idx_NxBx2 = tf.stack((idx, batch_NxB), axis=2)             # (n_particles, batch_size, 2)

        return resample_idx_NxBx2

    def reweight_log_Ws(self, log_Ws, all_fs):
        time = len(log_Ws)
        reweighted_log_Ws = ["placeholder"] * time

        for t in reversed(range(time)):
            if t == time - 1:
                reweighted_log_Ws[t] = log_Ws[t]
            else:
                denominator = \
                    tf.reduce_logsumexp(tf.expand_dims(log_Ws[t], axis=1) + all_fs[t], axis=0)
                reweight_factor = tf.reduce_logsumexp(reweighted_log_Ws[t + 1] +
                                                      all_fs[t] -
                                                      denominator,
                                                      axis=1)
                if self.use_stop_gradient:
                    reweighted_log_Ws[t] = log_Ws[t] + self.smoothing_perc * tf.stop_gradient(reweight_factor)
                else:
                    reweighted_log_Ws[t] = log_Ws[t] + self.smoothing_perc * reweight_factor

        return reweighted_log_Ws

    @staticmethod
    def compute_log_ZSMC(log_Ws):
        # log_Ws (list)
        # compute loss
        log_Ws = tf.stack(log_Ws)
        time, n_particles, batch_size = log_Ws.get_shape().as_list()

        log_ZSMC = tf.reduce_logsumexp(log_Ws, axis=1)
        log_ZSMC = tf.reduce_sum(log_ZSMC, name="log_ZSMC")
        log_ZSMC -= tf.log(tf.constant(n_particles, dtype=tf.float32)) * time * batch_size

        return log_ZSMC

    def n_step_MSE(self, n_steps, hidden, obs):
        # Compute MSE_k for k = 0, ..., n_steps
        batch_size, time, _, Dx = hidden.shape.as_list()
        batch_size, time, Dy = obs.shape.as_list()
        assert n_steps < time, "n_steps = {} >= time".format(n_steps)
        with tf.variable_scope(self.name):

            hidden = tf.reduce_mean(hidden, axis=2)
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

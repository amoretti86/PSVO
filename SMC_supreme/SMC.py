import tensorflow as tf
from tensorflow_probability import distributions as tfd
from tensorflow.contrib.layers import fully_connected, xavier_initializer


class SMC:
    def __init__(self, q0, q1, q2, f, g,
                 n_particles,
                 q_uses_true_X=False,
                 bRNN=None,
                 attention_encoder=None,
                 X0_use_separate_RNN=True,
                 use_stack_rnn=False,
                 FFBS=False,
                 smoothing_perc=1.0,
                 FFBS_to_learn=False,
                 use_stop_gradient=False,
                 use_input=False,
                 name="log_ZSMC"):

        self.q0 = q0
        self.q1 = q1
        self.q2 = q2
        self.f = f
        self.g = g
        self.n_particles = n_particles

        self.q_uses_true_X = q_uses_true_X

        if (bRNN or attention_encoder) is not None:
            self.smooth_obs = True
            self.X0_use_separate_RNN = X0_use_separate_RNN
            self.use_stack_rnn = use_stack_rnn

            assert not (bRNN is not None and attention_encoder is not None)

            if bRNN is not None:
                self.y_smoother_f, self.y_smoother_b, self.X0_smoother_f, self.X0_smoother_b = bRNN
                self.attention_encoder = None
                if not use_stack_rnn:
                    self.y_smoother_f = tf.nn.rnn_cell.MultiRNNCell(self.y_smoother_f)
                    self.y_smoother_b = tf.nn.rnn_cell.MultiRNNCell(self.y_smoother_b)
                    if X0_use_separate_RNN:
                        self.X0_smoother_f = tf.nn.rnn_cell.MultiRNNCell(self.X0_smoother_f)
                        self.X0_smoother_b = tf.nn.rnn_cell.MultiRNNCell(self.X0_smoother_b)
            if attention_encoder is not None:
                self.attention_encoder = attention_encoder
        else:
            self.smooth_obs = False

        self.use_stop_gradient = use_stop_gradient
        self.use_input = use_input

        self.FFBS = FFBS
        self.smoothing_perc = smoothing_perc
        self.FFBS_to_learn = FFBS_to_learn
        if not self.FFBS_to_learn:
            self.smoothing_perc = 1

        self.name = name

    def get_log_ZSMC(self, obs, hidden, Input, q_cov=1):
        """
        Get log_ZSMC from obs y_1:T
        Input:
            obs.shape = (batch_size, time, Dy)
            hidden.shape = (batch_size, time, Dz)
            Input.shape = (batch_size, time, Di)
        Output:
            log_ZSMC: shape = scalar
            log: stuff to debug
        """
        with tf.variable_scope(self.name):
            batch_size, time, Dy = obs.get_shape().as_list()
            n_particles = self.n_particles

            self.n_particles, self.batch_size, self.time, self.Dy = n_particles, batch_size, time, Dy

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

            if self.smooth_obs:
                X_ancestor, smoothed_obs = self._smooth_obs(obs)
            else:
                smoothed_obs = tf.unstack(obs, axis=1)
                X_ancestor = smoothed_obs[0]

            for t in range(0, time):
                if t == 0:
                    sample_size = (n_particles)
                else:
                    sample_size = ()

                if self.use_input:
                    if t == 0:
                        if X_ancestor is None:
                            q_f_t_feed = Input[:, 0, :]
                        else:
                            q_f_t_feed = tf.concat([X_ancestor, Input[:, 0, :]], axis=-1)
                    else:
                        Input_t_expanded = tf.tile(tf.expand_dims(Input[:, t, :], axis=0), (n_particles, 1, 1))
                        q_f_t_feed = tf.concat([X_ancestor, Input_t_expanded], axis=-1)
                else:
                    q_f_t_feed = X_ancestor

                if self.q_uses_true_X:
                    Dx = hidden.get_shape().as_list()[-1]
                    mvn = tfd.MultivariateNormalFullCovariance(hidden[:, t, :], q_cov * tf.eye(Dx),
                                                               name="q_{}_mvn".format(t))
                    X = mvn.sample((n_particles))
                    q_t_log_prob = mvn.log_prob(X)
                else:
                    if self.q2 is None:
                        if t == 0:
                            X, q_t_log_prob = self.q0.sample_and_log_prob(q_f_t_feed, sample_shape=sample_size,
                                                                          name="q_{}_sample_and_log_prob".format(t))
                        else:
                            X, q_t_log_prob = self.q1.sample_and_log_prob(q_f_t_feed, sample_shape=sample_size,
                                                                          name="q_{}_sample_and_log_prob".format(t))
                    else:
                        if t == 0:
                            X, q_t_log_prob, f_t_log_prob = self.sample_from_2_dist(self.q0, self.q2,
                                                                                    q_f_t_feed, smoothed_obs[t],
                                                                                    sample_size)
                        else:
                            X, q_t_log_prob, f_t_log_prob = self.sample_from_2_dist(self.q1, self.q2,
                                                                                    q_f_t_feed, smoothed_obs[t],
                                                                                    sample_size)

                if self.FFBS and t != 0:
                    X_tile = tf.tile(tf.expand_dims(X, axis=1), (1, n_particles, 1, 1), name="X_tile")

                    if self.use_input:
                        Input_t_expanded = tf.tile(tf.expand_dims(Input[:, t, :], axis=0), (n_particles, 1, 1))
                        f_t_all_feed = tf.concat([X_prev, Input_t_expanded], axis=-1)
                    else:
                        f_t_all_feed = X_prev

                    f_t_all_log_prob = self.f.log_prob(f_t_all_feed, X_tile, name="f_{}_all_log_prob".format(t))
                    all_fs.append(f_t_all_log_prob)

                    f_t_log_prob = self.f.log_prob(q_f_t_feed, X, name="f_{}_log_prob".format(t))

                elif self.f != self.q1 and t != 0:
                    f_t_log_prob = self.f.log_prob(q_f_t_feed, X, name="f_{}_log_prob".format(t))
                else:
                    # if f and q1 share the same network, f_t_log_prob is already calculated in self.sample_from_2_dist
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

                if n_particles > 1:
                    resample_idx = self.get_resample_idx(log_W, t)
                    X_ancestor = tf.gather_nd(X, resample_idx)                    # (n_particles, batch_size, Dx)
                else:
                    X_ancestor = X
                X_prev = X

                # collect X after rather than before resampling
                X_prevs.append(X_prev)
                X_ancestors.append(X_ancestor)

            # to make sure len(Xs) = time
            X_prevs.append(X_prev)
            X_ancestors.append(X_ancestor)

            reweighted_log_Ws = None
            if self.FFBS:
                reweighted_log_Ws = self.reweight_log_Ws(log_Ws, all_fs)
                if self.FFBS_to_learn:
                    log_ZSMC = self.compute_log_ZSMC(reweighted_log_Ws)
                else:
                    log_ZSMC = self.compute_log_ZSMC(log_Ws)

                Xs = []
                for t in range(time):
                    smoothing_idx = self.get_resample_idx(reweighted_log_Ws[t], t)
                    smoothed_X_t = tf.gather_nd(X_prevs[t], smoothing_idx)
                    Xs.append(smoothed_X_t)

                reweighted_log_Ws = tf.stack(reweighted_log_Ws)
                reweighted_log_Ws = tf.transpose(reweighted_log_Ws, perm=[2, 0, 1], name="reweighted_log_Ws")
                log = [reweighted_log_Ws]
            else:
                log_ZSMC = self.compute_log_ZSMC(log_Ws)
                Xs = X_ancestors


            # (batch_size, time, n_particles, Dx)
            Xs = tf.transpose(tf.stack(Xs), perm=[2, 0, 1, 3], name="Xs")
            X_prevs = tf.transpose(tf.stack(X_prevs), perm=[2, 0, 1, 3], name="X_prev")
            X_ancestors = tf.transpose(tf.stack(X_ancestors), perm=[2, 0, 1, 3], name="X_ancestors")

            log_Ws = tf.transpose(tf.stack(log_Ws), perm=[2, 0, 1], name="log_Ws")     # (batch_size, time, n_particles)
            qs = tf.transpose(tf.stack(qs), perm=[2, 0, 1], name="qs")                 # (batch_size, time, n_particles)
            fs = tf.transpose(tf.stack(fs), perm=[2, 0, 1], name="fs")                 # (batch_size, time, n_particles)
            gs = tf.transpose(tf.stack(gs), perm=[2, 0, 1], name="gs")                 # (batch_size, time, n_particles)

            log = {"Xs": Xs,
                   "X_prevs": X_prevs,
                   "X_ancestors": X_ancestors,
                   "log_Ws": log_Ws,
                   "reweighted_log_Ws": reweighted_log_Ws,
                   "qs": qs,
                   "fs": fs,
                   "gs": gs}

        return log_ZSMC, log

    def sample_from_2_dist(self, dist1, dist2, d1_input, d2_input, sample_size):
        d1_mvn = dist1.get_mvn(d1_input)
        d2_mvn = dist2.get_mvn(d2_input)

        if isinstance(d1_mvn, tfd.MultivariateNormalDiag) and isinstance(d2_mvn, tfd.MultivariateNormalDiag):
            d1_mvn_mean, d1_mvn_cov = d1_mvn.mean(), d1_mvn.stddev()
            d2_mvn_mean, d2_mvn_cov = d2_mvn.mean(), d2_mvn.stddev()

            d1_mvn_cov_inv, d2_mvn_cov_inv = 1 / d1_mvn_cov, 1 / d2_mvn_cov
            combined_cov = 1 / (d1_mvn_cov_inv + d2_mvn_cov_inv)
            combined_mean = combined_cov * (d1_mvn_cov_inv * d1_mvn_mean + d2_mvn_cov_inv * d2_mvn_mean)

            mvn = tfd.MultivariateNormalDiag(combined_mean,
                                             combined_cov,
                                             validate_args=True,
                                             allow_nan_stats=False)
        else:
            if isinstance(d1_mvn, tfd.MultivariateNormalDiag):
                d1_mvn_mean, d1_mvn_cov = d1_mvn.mean(), tf.diag(d1_mvn.stddev())
            else:
                d1_mvn_mean, d1_mvn_cov = d1_mvn.mean(), d1_mvn.covariance()
            if isinstance(d2_mvn, tfd.MultivariateNormalDiag):
                d2_mvn_mean, d2_mvn_cov = d2_mvn.mean(), tf.diag(d2_mvn.stddev())
            else:
                d2_mvn_mean, d2_mvn_cov = d2_mvn.mean(), d2_mvn.covariance()

            if len(d1_mvn_cov.shape.as_list()) == 2:
                d1_mvn_cov = tf.expand_dims(d1_mvn_cov, axis=0)

            d1_mvn_cov_inv, d2_mvn_cov_inv = tf.linalg.inv(d1_mvn_cov), tf.linalg.inv(d2_mvn_cov)
            combined_cov = tf.linalg.inv(d1_mvn_cov_inv + d2_mvn_cov_inv)
            perm = list(range(len(combined_cov.shape)))
            perm[-2], perm[-1] = perm[-1], perm[-2]
            combined_cov = (combined_cov + tf.transpose(combined_cov, perm=perm)) / 2
            combined_mean = tf.matmul(combined_cov,
                                      tf.matmul(d1_mvn_cov_inv, tf.expand_dims(d1_mvn_mean, axis=-1)) +
                                      tf.matmul(d2_mvn_cov_inv, tf.expand_dims(d2_mvn_mean, axis=-1))
                                      )
            combined_mean = tf.squeeze(combined_mean, axis=-1)

            mvn = tfd.MultivariateNormalFullCovariance(combined_mean,
                                                       combined_cov,
                                                       validate_args=True,
                                                       allow_nan_stats=False)

        X = mvn.sample(sample_size)
        q_t_log_prob = mvn.log_prob(X)
        f_t_log_prob = d1_mvn.log_prob(X)

        return X, q_t_log_prob, f_t_log_prob

    def get_resample_idx(self, log_W, t):
        # get resample index a_t^k ~ Categorical(w_t^1, ..., w_t^K)
        n_particles, batch_size = self.n_particles, self.batch_size

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
        # reweight weight of each particle (w_t^k) according to FFBS formula
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
        # log_Ws: list of log_W for t = 1, ..., T
        # compute loss
        log_Ws = tf.stack(log_Ws)
        time, n_particles, batch_size = log_Ws.get_shape().as_list()

        log_ZSMC = tf.reduce_logsumexp(log_Ws, axis=1)
        log_ZSMC = tf.reduce_sum(log_ZSMC, name="log_ZSMC")
        log_ZSMC -= tf.log(tf.constant(n_particles, dtype=tf.float32)) * time * batch_size

        return log_ZSMC

    def _smooth_obs(self, obs):
        # smooth obs with bidirectional RNN or attention mechanism

        with tf.variable_scope("smooth_obs"):
            if self.attention_encoder is None:

                if self.use_stack_rnn:
                    outputs, state_fw, state_bw = \
                        tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.y_smoother_f,
                                                                       self.y_smoother_b,
                                                                       obs,
                                                                       dtype=tf.float32)
                else:
                    outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(self.y_smoother_f,
                                                                                    self.y_smoother_b,
                                                                                    obs,
                                                                                    dtype=tf.float32)
                smoothed_obs = tf.concat(outputs, axis=-1)

                if self.X0_use_separate_RNN:
                    if self.use_stack_rnn:
                        outputs, state_fw, state_bw = \
                            tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.X0_smoother_f,
                                                                           self.X0_smoother_b,
                                                                           obs,
                                                                           dtype=tf.float32)
                    else:
                        outputs, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(self.X0_smoother_f,
                                                                                        self.X0_smoother_b,
                                                                                        obs,
                                                                                        dtype=tf.float32)

                smoothed_X0 = tf.concat([tf.concat(state, axis=-1) for state in state_fw + state_bw], axis=-1)

            else:
                smoothed_obs = self.attention_encoder(obs)
                smoothed_obs_list = tf.unstack(smoothed_obs, axis=1)
                smoothed_X0 = smoothed_obs_list[0]

            smoothed_obs_list = tf.unstack(smoothed_obs, axis=1)

        return smoothed_X0, smoothed_obs_list

    def n_step_MSE(self, n_steps, hidden, obs, Input):
        # Compute MSE_k for k = 0, ..., n_steps
        # Intermediate step to calculate k-step R^2
        batch_size, time, _, _ = hidden.shape.as_list()
        _, _, Dy = obs.shape.as_list()
        _, _, Di = Input.shape.as_list()
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
                if self.use_input:
                    f_k_input = tf.concat((x_BxTmkxDz, Input[:, :-(k + 1), :]), axis=-1)
                else:
                    f_k_input = x_BxTmkxDz
                x_BxTmkxDz = self.f.mean(f_k_input)                                # (batch_size, time - k - 1, Dx)

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
        # only used for drawing 2D quiver plot
        if self.use_input:
            return None
        with tf.variable_scope(self.name):
            return self.f.mean(X)

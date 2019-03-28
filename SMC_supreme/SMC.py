import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class SMC:
    def __init__(self, model,
                 n_particles,
                 q_uses_true_X=False,
                 use_input=False,
                 X0_use_separate_RNN=True,
                 use_stack_rnn=False,
                 FFBS=False,
                 FFBS_to_learn=False,
                 TFS=False,
                 name="log_ZSMC"):

        self.model = model

        # SSM distributions
        self.q0 = model.q0_dist
        self.q1 = model.q1_dist
        self.q2 = model.q2_dist
        self.f  = model.f_dist
        self.g  = model.g_dist

        if TFS:
            self.q1_inv = model.q1_inv_dist
            self.f_inv = model.f_inv_dist
            if model.TFS_use_diff_q0:
                self.TFS_q0 = model.TFS_q0_dist

        self.n_particles = n_particles

        self.q_uses_true_X = q_uses_true_X
        self.use_input = use_input

        # bidirectional RNN as full sequence observations encoder
        self.X0_use_separate_RNN = X0_use_separate_RNN
        self.use_stack_rnn = use_stack_rnn

        if model.bRNN is not None:
            self.smooth_obs = True
            self.y_smoother_f, self.y_smoother_b, self.X0_smoother_f, self.X0_smoother_b = model.bRNN
        else:
            self.smooth_obs = False

        # FFBS
        self.FFBS = FFBS
        self.FFBS_to_learn = FFBS_to_learn
        self.smoothing_perc = model.smoothing_perc
        if not self.FFBS_to_learn:
            self.smoothing_perc = 1

        # TFS
        self.TFS = TFS

        assert not (FFBS and TFS), "cannot use FFBS and TFS at the same time"

        self.name = name

    def get_log_ZSMC(self, obs, hidden, Input):
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
        n_particles = self.n_particles
        batch_size, time, _ = obs.get_shape().as_list()
        self.Dx, self.batch_size, self.time = self.model.Dx, batch_size, time

        with tf.variable_scope(self.name):

            # get X_1:T, resampled X_1:T and log(W_1:T) from SMC
            X_prevs, X_ancestors, log_Ws = self.SMC(Input, hidden, obs, forward=True)

            # TFS
            if self.TFS:
                X_prevs_b, X_ancestors_b, log_Ws_b = self.SMC(Input, hidden, obs, forward=False)
                log_Ws = self.TFS_reweight_log_Ws(log_Ws, log_Ws_b, X_prevs, X_prevs_b, Input)
                X_ancestors = [self.resample_X(X, log_W, sample_size=n_particles)
                               for X, log_W in zip(tf.unstack(X_prevs_b), tf.unstack(log_Ws))]
                X_ancestors = tf.stack(X_ancestors)

            # FFBS
            if self.FFBS:
                reweighted_log_Ws = self.FFBS_reweight_log_Ws(log_Ws, X_prevs, Input)
                X_ancestors = [self.resample_X(X, log_W, sample_size=n_particles)
                               for X, log_W in zip(tf.unstack(X_prevs), tf.unstack(reweighted_log_Ws))]
                X_ancestors = tf.stack(X_ancestors)
                if self.FFBS_to_learn:
                    log_Ws = reweighted_log_Ws

            # compute log_ZSMC
            log_ZSMC = self.compute_log_ZSMC(log_Ws)
            Xs = X_ancestors

            # shape = (batch_size, time, n_particles, Dx)
            Xs = tf.transpose(Xs, perm=[2, 0, 1, 3], name="Xs")

            log = {"Xs": Xs}

        return log_ZSMC, log

    def SMC(self, Input, hidden, obs, forward=True, q_cov=1.0):
        Dx, time, n_particles, batch_size = self.Dx, self.time, self.n_particles, self.batch_size

        # preprossing obs
        preprocessed_X0, preprocessed_obs = self.preprocess_obs(obs, forward)
        if forward:
            q0 = self.q0
            q1 = self.q1
            f  = self.f
        else:
            Input  = Input[:, ::-1, :]
            hidden = hidden[:, ::-1, :]
            obs    = obs[:, ::-1, :]
            q0 = self.TFS_q0 if self.model.TFS_use_diff_q0 else self.q0
            q1 = self.q1_inv
            f  = self.f_inv

        # t = 0
        q_f_t_feed = preprocessed_X0
        if self.use_input:
            q_f_t_feed = tf.concat([q_f_t_feed, Input[:, 0, :]], axis=-1)

        if self.q_uses_true_X:
            X, q_t_log_prob = self.sample_from_true_X(hidden[:, 0, :],
                                                      q_cov,
                                                      sample_shape=n_particles,
                                                      name="q_{}_sample_and_log_prob".format(0))
        else:
            if self.model.use_2_q:
                X, q_t_log_prob, f_t_log_prob = self.sample_from_2_dist(q0,
                                                                        self.q2,
                                                                        q_f_t_feed,
                                                                        preprocessed_obs[0],
                                                                        sample_size=n_particles)
            else:
                X, q_t_log_prob = q0.sample_and_log_prob(q_f_t_feed,
                                                         sample_shape=n_particles,
                                                         name="q_{}_sample_and_log_prob".format(0))

        # only when use_bootstrap and use_2_q, f_t_log_prob has been calculated
        if not (self.model.use_bootstrap and self.model.use_2_q):
            f_t_log_prob = f.log_prob(q_f_t_feed, X, name="f_{}_log_prob".format(0))

        g_t_log_prob = self.g.log_prob(X, obs[:, 0], name="g_{}_log_prob".format(0))
        log_W = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_W_{}".format(0))

        # t = 1, ..., T - 1
        # prepare tensor arrays
        preprocessed_obs_ta = \
            tf.TensorArray(tf.float32, size=time, name="preprocessed_obs_ta").unstack(preprocessed_obs)
        log_names = ["X_prevs", "X_ancestors", "log_Ws"]
        log = [tf.TensorArray(tf.float32, size=time, name="{}_ta".format(name)) for name in log_names]
        log[2] = log[2].write(0, log_W)

        def while_cond(t, *unused_args):
            return t < time

        def while_body(t, X_prev, log_W, log):
            X_ancestor = self.resample_X(X_prev, log_W, sample_size=n_particles)
            q_f_t_feed = X_ancestor

            if self.use_input:
                Input_t_expanded = tf.tile(tf.expand_dims(Input[:, t, :], axis=0), (n_particles, 1, 1))
                q_f_t_feed = tf.concat([q_f_t_feed, Input_t_expanded], axis=-1)

            if self.q_uses_true_X:
                X, q_t_log_prob = self.sample_from_true_X(hidden[:, t, :],
                                                          q_cov,
                                                          sample_shape=(),
                                                          name="q_t_sample_and_log_prob")
            else:
                if self.model.use_2_q:
                    X, q_t_log_prob, f_t_log_prob = self.sample_from_2_dist(q1,
                                                                            self.q2,
                                                                            q_f_t_feed,
                                                                            preprocessed_obs_ta.read(t),
                                                                            sample_size=())
                else:
                    X, q_t_log_prob = q1.sample_and_log_prob(q_f_t_feed,
                                                             sample_shape=(),
                                                             name="q_t_sample_and_log_prob")

            if not (self.model.use_bootstrap and self.model.use_2_q):
                f_t_log_prob = f.log_prob(q_f_t_feed, X, name="f_t_log_prob")

            g_t_log_prob = self.g.log_prob(X, obs[:, t], name="g_t_log_prob")
            log_W = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_W_t")

            # write to tensor arrays
            idxs = [t - 1, t - 1, t]
            log_contents = [X_prev, X_ancestor, log_W]
            log = [ta.write(idx, log_content) for ta, idx, log_content in zip(log, idxs, log_contents)]

            return (t + 1, X, log_W, log)

        # conduct the while loop
        init_state = (1, X, log_W, log)
        t_final, X_T, log_W, log = tf.while_loop(while_cond, while_body, init_state)

        # write final results at t = T - 1 to tensor arrays
        X_T_resampled = self.resample_X(X_T, log_W, sample_size=n_particles)
        log_contents = [X_T, X_T_resampled, log_W]
        log = [ta.write(t_final - 1, log_content) if i != 2 else ta
               for ta, log_content, i in zip(log, log_contents, range(3))]

        # transfer tensor arrays to tensors
        log_shapes = [(time, n_particles, batch_size, Dx)] * 2 + [(time, n_particles, batch_size)]
        log = [ta.stack(name=name) for ta, name in zip(log, log_names)]
        for tensor, shape in zip(log, log_shapes):
            tensor.set_shape(shape)

        if not forward:
            log = [tf.stack(list(reversed(tf.unstack(tensor, axis=0)))) for tensor in log]

        X_prevs, X_ancestors, log_Ws = log

        return X_prevs, X_ancestors, log_Ws

    def sample_from_2_dist(self, dist1, dist2, d1_input, d2_input, sample_size=()):
        d1_mvn = dist1.get_mvn(d1_input)
        d2_mvn = dist2.get_mvn(d2_input)

        can_solve_analytically = True
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
            elif isinstance(d1_mvn, tfd.MultivariateNormalFullCovariance):
                d1_mvn_mean, d1_mvn_cov = d1_mvn.mean(), d1_mvn.covariance()
            else:
                can_solve_analytically = False
            if isinstance(d2_mvn, tfd.MultivariateNormalDiag):
                d2_mvn_mean, d2_mvn_cov = d2_mvn.mean(), tf.diag(d2_mvn.stddev())
            elif isinstance(d2_mvn, tfd.MultivariateNormalFullCovariance):
                d2_mvn_mean, d2_mvn_cov = d2_mvn.mean(), d2_mvn.covariance()
            else:
                can_solve_analytically = False

            if can_solve_analytically:
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
        if can_solve_analytically:
            X = mvn.sample(sample_size)
            q_t_log_prob = mvn.log_prob(X)
            f_t_log_prob = d1_mvn.log_prob(X)
        else:
            assert sample_size == ()
            X = d1_mvn.sample(dist1.transformation.sample_num)
            f_t_log_prob   = d1_mvn.log_prob(X)
            q2_t_log_prob  = d2_mvn.log_prob(X)
            aggr_log_prob  = f_t_log_prob + q2_t_log_prob
            f_t_log_prob  -= tf.reduce_logsumexp(f_t_log_prob, axis=0)
            aggr_log_prob -= tf.reduce_logsumexp(aggr_log_prob, axis=0)
            X, q_t_log_prob, f_t_log_prob = self.resample_X([X, aggr_log_prob, f_t_log_prob],
                                                            aggr_log_prob,
                                                            sample_size=())
        return X, q_t_log_prob, f_t_log_prob

    def sample_from_true_X(self, hidden, q_cov, sample_shape=(), name="q_t_mvn"):
        mvn = tfd.MultivariateNormalDiag(hidden,
                                         q_cov * tf.ones(self.Dx, dtype=tf.float32),
                                         name=name)
        X = mvn.sample(sample_shape)
        q_t_log_prob = mvn.log_prob(X)

        return X, q_t_log_prob

    def resample_X(self, X, log_W, sample_size=()):
        if log_W.shape.as_list()[0] != 1:
            resample_idx = self.get_resample_idx(log_W, sample_size)
            if isinstance(X, list):
                X_resampled = [tf.gather_nd(item, resample_idx) for item in X]
            else:
                X_resampled = tf.gather_nd(X, resample_idx)
        else:
            assert sample_size == 1
            X_resampled = X

        return X_resampled

    def get_resample_idx(self, log_W, sample_size=()):
        # get resample index a_t^k ~ Categorical(w_t^1, ..., w_t^K)
        nb_classes  = log_W.shape.as_list()[0]
        batch_shape = log_W.shape.as_list()[1:]
        perm = list(range(1, len(batch_shape) + 1)) + [0]

        log_W_max = tf.stop_gradient(tf.reduce_max(log_W, axis=0))
        log_W = tf.transpose(log_W - log_W_max, perm=perm)
        categorical = tfd.Categorical(logits=log_W, validate_args=True, name="Categorical")

        # sample multiple times to remove idx out of range
        if sample_size == ():
            idx_shape = batch_shape
        else:
            assert isinstance(sample_size, int), "sample_size should be int, {} is given".format(sample_size)
            idx_shape = [sample_size] + batch_shape

        idx = tf.ones(idx_shape, dtype=tf.int32) * nb_classes
        for _ in range(1):
            fixup_idx = categorical.sample(sample_size)
            idx = tf.where(idx >= nb_classes, fixup_idx, idx)

        # if still got idx out of range, replace them with idx from uniform distribution
        final_fixup = tf.random.uniform(idx_shape, maxval=nb_classes, dtype=tf.int32)
        idx = tf.where(idx >= nb_classes, final_fixup, idx)

        batch_idx = np.meshgrid(*[range(i) for i in idx_shape], indexing='ij')
        if sample_size != ():
            batch_idx = batch_idx[1:]
        resample_idx = tf.stack([idx] + batch_idx, axis=-1)

        return resample_idx

    def TFS_reweight_log_Ws(self, log_Ws, log_Ws_b, X_prevs, X_prevs_b, Input):
        # reweight weight of each particle (w_t^k) according to TFS formula9
        n_particles, time = self.n_particles, self.time

        log_Ws,  log_Ws_b  = tf.unstack(log_Ws),  tf.unstack(log_Ws_b)
        X_prevs, X_prevs_b = tf.unstack(X_prevs), tf.unstack(X_prevs_b)

        all_fs            = ["placeholder"] * time
        reweighted_log_Ws = ["placeholder"] * time

        for t in range(time - 1):
            X_tile = tf.tile(tf.expand_dims(X_prevs_b[t + 1], axis=1), (1, n_particles, 1, 1), name="X_tile")

            f_t_all_feed = X_prevs[t]
            if self.use_input:
                Input_t_expanded = tf.tile(tf.expand_dims(Input[:, t, :], axis=0), (n_particles, 1, 1))
                f_t_all_feed = tf.concat([f_t_all_feed, Input_t_expanded], axis=-1)

            # all_fs[t, i, j] = f(tilde{x}_{t + 1}^i | x_t^j)
            all_fs[t] = self.f.log_prob(f_t_all_feed, X_tile, name="f_t_all_log_prob")

        for t in range(time):
            if t == 0:
                reweighted_log_Ws[t] = log_Ws_b[t]
            else:
                reweight_factor = tf.reduce_logsumexp(log_Ws[t - 1] + all_fs[t - 1], axis=1)
                reweighted_log_Ws[t] = log_Ws_b[t] + reweight_factor

        reweighted_log_Ws = tf.stack(reweighted_log_Ws)
        return reweighted_log_Ws

    def FFBS_reweight_log_Ws(self, log_Ws, X_prevs, Input):
        # reweight weight of each particle (w_t^k) according to FFBS formula
        n_particles, time = self.n_particles, self.time

        log_Ws, X_prevs = tf.unstack(log_Ws), tf.unstack(X_prevs)

        all_fs            = ["placeholder"] * time
        reweighted_log_Ws = ["placeholder"] * time

        for t in range(time - 1):
            X_tile = tf.tile(tf.expand_dims(X_prevs[t + 1], axis=1), (1, n_particles, 1, 1), name="X_tile")

            f_t_all_feed = X_prevs[t]
            if self.use_input:
                Input_t_expanded = tf.tile(tf.expand_dims(Input[:, t, :], axis=0), (n_particles, 1, 1))
                f_t_all_feed = tf.concat([f_t_all_feed, Input_t_expanded], axis=-1)

            # all_fs[t, i, j] = f(x_{t + 1}^i | x_t^j)
            all_fs[t] = self.f.log_prob(f_t_all_feed, X_tile, name="f_t_all_log_prob")

        for t in reversed(range(time)):
            if t == time - 1:
                reweighted_log_Ws[t] = log_Ws[t]
            else:
                denominator = tf.reduce_logsumexp(log_Ws[t] + all_fs[t], axis=1)
                reweight_factor = tf.reduce_logsumexp(
                    tf.expand_dims(reweighted_log_Ws[t + 1] - denominator, axis=1) + all_fs[t],
                    axis=0)
                reweighted_log_Ws[t] = log_Ws[t] + self.smoothing_perc * reweight_factor

        reweighted_log_Ws = tf.stack(reweighted_log_Ws)
        return reweighted_log_Ws

    @staticmethod
    def compute_log_ZSMC(log_Ws):
        # log_Ws: tensor of log_W for t = 0, ..., T - 1
        # compute loss
        time, n_particles, batch_size = log_Ws.get_shape().as_list()

        log_ZSMC = tf.reduce_logsumexp(log_Ws, axis=1)
        log_ZSMC = tf.reduce_sum(log_ZSMC, name="log_ZSMC")
        log_ZSMC -= tf.log(tf.constant(n_particles, dtype=tf.float32)) * time * batch_size

        return log_ZSMC

    def preprocess_obs(self, obs, forward=True):
        # if self.smooth_obs, smooth obs with bidirectional RNN

        with tf.variable_scope("smooth_obs"):
            if not self.smooth_obs:
                preprocessed_obs = tf.unstack(obs, axis=1)
                if not forward:
                    preprocessed_obs = list(reversed(preprocessed_obs))
                preprocessed_X0 = preprocessed_obs[0]

            else:
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
                preprocessed_obs = tf.unstack(smoothed_obs, axis=1)

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
                if self.use_stack_rnn:
                    outputs_fw = outputs_bw = outputs
                else:
                    outputs_fw, outputs_bw = outputs
                output_fw_list, output_bw_list = tf.unstack(outputs_fw, axis=1), tf.unstack(outputs_bw, axis=1)
                preprocessed_X0 = tf.concat([output_fw_list[-1], output_bw_list[0]], axis=-1)

                if not forward:
                    preprocessed_obs = list(reversed(preprocessed_obs))
                    preprocessed_X0 = tf.concat([output_bw_list[0], output_fw_list[-1]], axis=-1)

            if not (self.model.use_bootstrap and self.model.use_2_q):
                preprocessed_X0 = self.model.X0_transformer(preprocessed_X0)

        return preprocessed_X0, preprocessed_obs

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
                x_BxTmkxDz = self.f.mean(f_k_input)                                 # (batch_size, time - k - 1, Dx)

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

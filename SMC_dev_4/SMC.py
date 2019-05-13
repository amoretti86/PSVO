import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd


class SMC:
    def __init__(self, model,
                 n_particles,
                 q_uses_true_X=False,
                 use_input=False,
                 IWAE=False,
                 X0_use_separate_RNN=True,
                 use_stack_rnn=False,
                 FFBS=False,
                 FFBS_loss_type='vae',
                 FFBS_particles=0,
                 FFBS_to_learn=False,
                 name="log_ZSMC"):

        self.model = model

        # SSM distributions
        self.q0 = model.q0_dist
        self.q1 = model.q1_dist
        self.q2 = model.q2_dist
        self.f  = model.f_dist
        self.g  = model.g_dist

        self.n_particles = n_particles

        self.q_uses_true_X = q_uses_true_X
        self.use_input = use_input

        # IWAE
        self.IWAE = IWAE

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
        self.FFBS_loss_type = FFBS_loss_type
        self.FFBS_particles = FFBS_particles

        if self.FFBS:
            assert self.FFBS_particles > 0

        self.FFBS_to_learn = FFBS_to_learn
        self.smoothing_perc = model.smoothing_perc
        if not self.FFBS_to_learn:
            self.smoothing_perc = 1

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
        Dx = self.Dx

        with tf.variable_scope(self.name):

            # FFBS
            if self.FFBS:
                bw_Xs = []
                FFBS_log_weights = []
                M = self.FFBS_particles
                for i in range(M):
                    bw_X, FFBS_log_weight = self.get_one_FFBS_pass(Input, hidden, obs)
                    bw_Xs.append(bw_X)
                    FFBS_log_weights.append(FFBS_log_weight)

                bw_Xs = tf.stack(bw_Xs)  # (M, time, batch_size, Dx)
                assert bw_Xs.shape.as_list() == [M, time, batch_size, Dx]
                bw_Xs = tf.transpose(bw_Xs, perm=[1, 0, 2, 3])
                assert bw_Xs.shape.as_list() == [time, M, batch_size, Dx]

                FFBS_log_weights = tf.stack(FFBS_log_weights)  # (M, time, batch_size)
                assert FFBS_log_weights.shape.as_list() == [M, time, batch_size]
                FFBS_log_weights = tf.transpose(FFBS_log_weights, perm=[1, 0, 2])
                assert FFBS_log_weights.shape.as_list() == [time, M, batch_size]

                log_ZSMC = self.compute_log_ZSMC(FFBS_log_weights)

                """
                if self.FFBS_loss_type == 'score':
                    log_ZSMC = self.compute_FFBS_score_loss(bw_Xs, obs)
                elif self.FFBS_loss_type == 'vae':
                    log_ZSMC = self.compute_FFBS_vae_loss(bw_Xs, obs, bw_log_weights)
                elif self.FFBS_loss_type == 'iwae':
                    log_ZSMC = self.compute_FFBS_iwae_loss(bw_Xs, obs, bw_log_weights)
                """
                Xs = bw_Xs

            else:
                # get X_1:T, resampled X_1:T and log(W_1:T) from SMC
                X_prevs, X_ancestors, log_Ws = self.SMC(Input, hidden, obs, forward=True)
                log_ZSMC = self.compute_log_ZSMC(log_Ws)
                Xs = X_ancestors

            # shape = (batch_size, time, n_particles or M, Dx)
            Xs = tf.transpose(Xs, perm=[2, 0, 1, 3], name="Xs")

            log = {"Xs": Xs}

        return log_ZSMC, log

    def get_one_FFBS_pass(self, Input, hidden, obs):
        """

        :param Input:
        :param hidden:
        :param obs:
        :return: one sample trajectory: bw_X -- (time, batch_size, Dx), and the loss,
        and the associated weight: bw_log_weight -- (batch_size, )
        """
        assert self.FFBS == True

        X_prevs, X_ancestors, log_Ws = self.SMC(Input, hidden, obs, forward=True)
        bw_Xs, FFBS_log_weights = self.FFBS_simulation(X_prevs, log_Ws, obs, M=1)  # shape (time, 1, batch_size, Dx)

        bw_X = tf.squeeze(bw_Xs, axis=1) # (time, batch_size, Dx)
        FFBS_log_weight = tf.squeeze(FFBS_log_weights, axis=1)  # (time, batch_size)

        return bw_X, FFBS_log_weight

    def FFBS_simulation(self, X, log_Ws, obs, M=1):
        """
        :param obs: (batch_size, time, Dy)
        :param X: shape (time, n_particles, batch_size, Dx)
        :param log_Ws: shape (time, n_particles, batch_size)
        :return: bw_Xs: M numebr of sample realizations, or the number of backward particles --(time, M, batch_size, Dx)
                FFBS_log_weights: M log weights associatied with M sample realizations --(time, M, batch_size)
        """

        time, n_particles, batch_size, Dx = X.shape.as_list()

        # store in reverse order
        bw_Xs_ta = tf.TensorArray(tf.float32, size=time, name="backward_X_ta")
        FFBS_log_weights_ta = tf.TensorArray(tf.float32, size=time, name='FFBS_log_weights_ta')

        # T-1
        # sample M particles from (n_particles, batch_size)
        f_Tminus1_log_prob = tf.constant(0.0, dtype=tf.float32, shape=(M, n_particles, batch_size),
                                         name='f_T-1_log_prob')
        bw_X_Tminus1, bw_log_weights_Tminus1, _, _ \
            = self.get_backward_sample_and_weight(X[time - 1], log_Ws[time - 1], f_Tminus1_log_prob, M,
                                                  last_timestep=True)

        bw_Xs_ta = bw_Xs_ta.write(time-1, bw_X_Tminus1)

        #  from T-2 to 0
        def while_cond(t, *unused_args):
            return t >= 0

        def while_body(t, bw_X_tplus1, bw_log_weights_tplus1, bw_Xs_ta, FFBS_log_weights_ta):
            bw_X_tplu1_for_f_eval = tf.expand_dims(bw_X_tplus1, 1)  # (M, 1, batch_size, Dx)
            bw_X_tplu1_for_f_eval = tf.tile(bw_X_tplu1_for_f_eval,
                                            [1, n_particles, 1, 1])  # (M, n_particles, batch_size, Dx)

            # X[t]: (n_particles, batch_size, Dx)
            # bw_X_tplus1_for_f_eval: (M, n_particles, batch_size, Dx)
            f_t_log_prob = self.f.log_prob(X[t], bw_X_tplu1_for_f_eval, name="f_t_log_prob")
            # f_t_log_prob: (M, n_particles, batch_size)

            bw_X_t, bw_log_weights_t, selected_fw_log_weights_t, selected_f_log_prob_tplus1 \
                = self.get_backward_sample_and_weight(X[t], log_Ws[t], f_t_log_prob, M)

            # X shape: (M, batch_size, Dx)
            # obs[:, t] shape: (batch_size, Dy)
            selected_g_log_prob_tplus1 = self.g.log_prob(bw_X_tplus1, obs[:, t+1])  # (M, batch_size)

            # p(x_t|y_1:t) f(x_t+1 | x_t) g(y_t+1 | x_t+1) / \tilde{w}_t+1 \tilde{w}_t
            FFBS_log_weights_t = \
                tf.add(selected_fw_log_weights_t + selected_f_log_prob_tplus1 + selected_g_log_prob_tplus1 \
                            - bw_log_weights_tplus1, - bw_log_weights_t, name="FFBS_log_weights_t")  # (M, batch_size)

            bw_Xs_ta = bw_Xs_ta.write(t, bw_X_t)
            FFBS_log_weights_ta = FFBS_log_weights_ta.write(t+1, FFBS_log_weights_t)

            return t - 1, bw_X_t, bw_log_weights_t, bw_Xs_ta, FFBS_log_weights_ta

        # conduct the while loop
        init_state = (time-2, bw_X_Tminus1, bw_log_weights_Tminus1, bw_Xs_ta, FFBS_log_weights_ta)
        t, bw_X_0, bw_log_weights_0, bw_Xs_ta, FFBS_log_weights_ta \
            = tf.while_loop(while_cond, while_body, init_state)

        # pre_X0: (batch_size, Dy)
        # bw_X_0: (M, batch_size, Dx)
        selected_f_log_prob_0 = self.q0.log_prob(self.pre_X0, bw_X_0, name="selected_f_log_prob_0")
        selected_g_log_prob_0 = self.g.log_prob(bw_X_0, obs[:, 0], name="selected_g_lor_prob_0")

        FFBS_log_weights_0 = tf.add(selected_f_log_prob_0 + selected_g_log_prob_0, - bw_log_weights_0,
                                    name="FFBS_log_weights_0")
        FFBS_log_weights_ta = FFBS_log_weights_ta.write(0, FFBS_log_weights_0)

        # transfer tensor arrays to tensors
        bw_Xs = bw_Xs_ta.stack()
        bw_Xs.set_shape((time, M, batch_size, Dx))

        FFBS_log_weights = FFBS_log_weights_ta.stack()
        FFBS_log_weights.set_shape((time, M, batch_size))

        return bw_Xs, FFBS_log_weights

    def _get_f_g(self, bw_Xs, obs):
        """
        sum over time
        :param bw_Xs: particles sampled from FFBSi, shape (time, M, batch_size, Dx)
        :param obs: observations, shape (batch_size, time, Dy)
        :return: f_log_prob, g_log_prob, each of shape (M, batch_size)
        """
        f_0_log_prob = self.q0.log_prob(self.pre_X0, bw_Xs[0], name="FFBS_f_0_log_prob")  # (M, batch_size)

        f_rest_log_prob = self.f.log_prob(bw_Xs[:-1], bw_Xs[1:],
                                          name="FFBS_f_rest_log_prob")  # (time-1, M,  batch_size)

        f_log_prob = tf.add(f_0_log_prob, tf.reduce_sum(f_rest_log_prob, axis=0), name="FFBS_log_prob")

        x_g_input = tf.transpose(bw_Xs, perm=(1, 2, 0, 3))  # (M, batch_size, time, Dx)
        g_log_prob = self.g.log_prob(x_g_input, obs)  # (M, batch_size, time)
        g_log_prob = tf.reduce_sum(g_log_prob, axis=2, name="FFBS_g_log_prob") # (M, batch_size)

        return f_log_prob, g_log_prob

    def compute_FFBS_score_loss(self, bw_Xs, obs):
        """
        :param bw_Xs: particles sampled from FFBSi, shape (time, M, batch_size, Dx)
        :param obs: observations, shape (batch_size, time, Dy)
        :return: 1 / M \sum_{i=1}^M p_\theta (x_{1:T}^{1:M}, y_{1:T}). shape ()
        """

        #assert self.pre_X0.shape == (batch_size, Dy)
        #assert bw_Xs[0].shape == (M, batch_size, Dx)

        f_log_prob, g_log_prob = self._get_f_g(bw_Xs, obs)  # each of shape (M, batch_size)

        f_log_prob_mean = tf.reduce_mean(f_log_prob)

        g_log_prob_mean = tf.reduce_mean(g_log_prob)

        score_loss = f_log_prob_mean + g_log_prob_mean

        return score_loss

    def compute_FFBS_vae_loss(self, bw_Xs, obs, bw_log_weights):
        """

        :param bw_Xs:
        :param obs:
        :param bw_log_weights: shape: (M, batch_size)
        :return: vae_loss = 1/M \sum_{m=1}^M [log p_m - log q_m],
        where log p_m = log f_m + log g_m ,and log q_m = log weight_m
        """
        score_loss = self.compute_FFBS_score_loss(bw_Xs, obs)
        assert len(score_loss.shape.as_list()) == 0

        vae_loss = score_loss - tf.reduce_mean(bw_log_weights)
        assert len(vae_loss.shape.as_list()) == 0

        return vae_loss

    def compute_FFBS_iwae_loss(self, bw_Xs, obs, bw_log_weights):
        """

        :param bw_Xs:
        :param obs:
        :param bw_log_weights:  shape: (M, batch_size)
        :return: iwae_loss = log {1/M \sum_{m=1}^M [p_m / q_m]  },
        where p_m / q_m = e^{log p_m - log q_m} = e^ {log f_m + log g_m - log weight_m}
        """
        M, _ = bw_log_weights.shape.as_list()

        f_log_prob, g_log_prob = self._get_f_g(bw_Xs, obs)  # each of shape (M, batch_size)
        p_over_q = f_log_prob + g_log_prob - bw_log_weights # (M, batch_size)
        iwae_loss = tf.reduce_logsumexp(p_over_q, axis=0) - tf.log(tf.cast(M, tf.float32))  # (batch_size, )
        iwae_loss = tf.reduce_mean(iwae_loss)  # over batch_size
        return iwae_loss

    @staticmethod
    def get_backward_sample_and_weight(X_t, log_W_t, f_t_log_prob, M, last_timestep=False):
        """
        :param X_t: shape(n_particles, batch_size, Dx)
        :param log_W_t: shape (n_particles, batch_size)
        :param f_t_log_prob: shape (M, n_particles, batch_size)
        :param M: number of the backward samples
        :return samples: M backward samples, shape (M, batch_size, Dx)
                log_bw_weights: associated weights with samples, shape (M, batch_size)
                selected_fw_log_weights: shape (M, batch_size)
                selected_f_log_prob: shape (M, batch_size)
                selected_g_log_prob: shape (M, batch_size)
        """
        _, batch_size, _ = X_t.shape.as_list()

        log_weights = log_W_t + f_t_log_prob  # (M, n_particles, batch_size)
        log_weights = tf.transpose(log_weights, perm=(2, 0, 1))  # (batch_size, M, n_particles)

        categorical = tfd.Categorical(logits=log_weights, validate_args=True, name="Categorical")

        sample_idx = categorical.sample()  # shape (batch_size, M)

        sample_idx = tf.transpose(sample_idx)  # (M, batch_size)
        sample_idx_reformat_1 = [[(sample_idx[i][j], j) for j in range(batch_size)] for i in range(M)]
        samples = tf.gather_nd(X_t, sample_idx_reformat_1, name="gather_particles")  # (M, batch_size, Dx)

        if last_timestep:
            selected_fw_log_weights = None
            selected_f_log_prob = None
        else:
            sample_idx_reformat_2 = [[(sample_idx[i][j], j) for j in range(batch_size)] for i in range(M)]
            selected_fw_log_weights = tf.gather_nd(log_W_t, sample_idx_reformat_2, name="gather_fw_log_weights")  # (M, batch_size)

            sample_idx_reformat_3 = [[(i, sample_idx[i][j], j) for j in range(batch_size)] for i in range(M)]
            selected_f_log_prob = tf.gather_nd(f_t_log_prob, sample_idx_reformat_3, name="gather_f_log_prob")

        # normalized version
        log_weights = tf.transpose(log_weights, perm=(1, 2, 0))  # (M, n_particles, batch_size)
        normalization_constant = tf.reduce_logsumexp(log_weights, axis=1)  # (M, batch_size)
        idx_reformat_4 = [[(i, sample_idx[i][j], j) for j in range(batch_size)] for i in range(M)]
        sample_log_unnormalized_weights = tf.gather_nd(log_weights, idx_reformat_4, name="gather_log_weights")  # (M, batch_size)
        sample_log_weights = sample_log_unnormalized_weights - normalization_constant

        return samples, sample_log_weights, selected_fw_log_weights, selected_f_log_prob

    def SMC(self, Input, hidden, obs, forward=True, q_cov=1.0):
        Dx, time, n_particles, batch_size = self.Dx, self.time, self.n_particles, self.batch_size

        # preprossing obs
        preprocessed_X0, preprocessed_obs = self.preprocess_obs(obs)
        self.pre_X0 = preprocessed_X0

        q0 = self.q0
        q1 = self.q1
        f = self.f

        # -------------------------------------- t = 0 -------------------------------------- #
        q_f_t_feed = preprocessed_X0
        if self.use_input:
            q_f_t_feed = tf.concat([q_f_t_feed, Input[:, 0, :]], axis=-1)

        # proposal
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
        # transition log probability
        # only when use_bootstrap and use_2_q, f_t_log_prob has been calculated
        if not (self.model.use_bootstrap and self.model.use_2_q):
            f_t_log_prob = f.log_prob(q_f_t_feed, X, name="f_{}_log_prob".format(0))

        # emission log probability and log weights
        g_t_log_prob = self.g.log_prob(X, obs[:, 0], name="g_{}_log_prob".format(0))

        log_alpha = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_alpha_{}".format(0))
        log_weight = tf.add(log_alpha, - tf.log(tf.constant(n_particles, dtype=tf.float32)),
                            name="log_weight_{}".format(0))  # (n_particles, batch_size)

        log_normalized_weight = tf.add(log_weight, - tf.reduce_logsumexp(log_weight, axis=0),
                                       name="log_noramlized_weight_{}".format(0))

        # -------------------------------------- t = 1, ..., T - 1 -------------------------------------- #
        # prepare tensor arrays
        preprocessed_obs_ta = \
            tf.TensorArray(tf.float32, size=time, name="preprocessed_obs_ta").unstack(preprocessed_obs)

        # tensor arrays to write
        log_names = ["X_prevs", "X_ancestors", "log_weight"]

        log = [tf.TensorArray(tf.float32, size=time, clear_after_read=False, name="{}_ta".format(name))
               for name in log_names]

        # write results for t = 0 into tensor arrays
        log[2] = log[2].write(0, log_weight)

        def while_cond(t, *unused_args):
            return t < time

        def while_body(t, X_prev, log_normalized_weight_tminus1, log):
            # resampling
            X_ancestor = self.resample_X(X_prev, log_normalized_weight_tminus1, sample_size=n_particles)

            q_f_t_feed = X_ancestor
            if self.use_input:
                Input_t_expanded = tf.tile(tf.expand_dims(Input[:, t, :], axis=0), (n_particles, 1, 1))
                q_f_t_feed = tf.concat([q_f_t_feed, Input_t_expanded], axis=-1)

            # proposal
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

            # transition log probability
            if not (self.model.use_bootstrap and self.model.use_2_q):
                f_t_log_prob = f.log_prob(q_f_t_feed, X, name="f_t_log_prob")

            # emission log probability and log weights
            g_t_log_prob = self.g.log_prob(X, obs[:, t], name="g_t_log_prob")

            log_alpha_t = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_alpha_t")
            # (n_particles, batch_size)

            log_weight_t = tf.add(log_alpha_t, log_normalized_weight_tminus1, name="log_weight_t")
            # (n_particles, batch_size)

            if self.IWAE:
                log_normalized_weight_t = tf.add(log_weight_t, - tf.reduce_logsumexp(log_weight_t, axis=0),
                                                 name="log_normalized_weight_t")
            else:
                log_normalized_weight_t = tf.negative(tf.log(tf.constant(n_particles, dtype=tf.float32,
                                                                         shape=(n_particles, batch_size),
                                                                         )), name="log_normalized_weight_t")

            # write results in this loop to tensor arrays
            idxs = [t - 1, t - 1, t]
            log_contents = [X_prev, X_ancestor, log_weight_t]
            log = [ta.write(idx, log_content) for ta, idx, log_content in zip(log, idxs, log_contents)]

            return (t + 1, X, log_normalized_weight_t, log)

        # conduct the while loop
        init_state = (1, X, log_normalized_weight, log)
        t_final, X_T, log_normalized_weight_T, log = tf.while_loop(while_cond, while_body, init_state)

        # write final results at t = T - 1 to tensor arrays
        X_T_resampled = self.resample_X(X_T, log_normalized_weight_T, sample_size=n_particles)
        log[0] = log[0].write(t_final - 1, X_T)
        log[1] = log[1].write(t_final - 1, X_T_resampled)

        # convert tensor arrays to tensors
        log_shapes = [(time, n_particles, batch_size, Dx)] * 2 + [(time, n_particles, batch_size)]

        log = [ta.stack(name=name) for ta, name in zip(log, log_names)]
        for tensor, shape in zip(log, log_shapes):
            tensor.set_shape(shape)

        X_prevs, X_ancestors, log_Ws = log

        return X_prevs, X_ancestors, log_Ws

    def sample_from_2_dist(self, dist1, dist2, d1_input, d2_input, sample_size=()):
        """

        :param dist1:
        :param dist2:
        :param d1_input:
        :param d2_input:
        :param sample_size:
        :return: X: (n_particles, batch_size, Dx), f_t_log_prob, q_t_log_prob: (n_particles, batch_size)
        """

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
        """
        :param X: ancestors to select from. shape: (n_particles, batch_size, Dx)
        :param log_W: log weights. shape (n_particles, batch_size)
        :param sample_size:
        :return: selected ancestors
        """
        if self.IWAE:
            X_resampled = X
            return X_resampled

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

    @staticmethod
    def compute_log_ZSMC(log_Ws):
        """
        :param log_Ws: shape (time, n_particles, batch_size)
        :return: loss, shape ()
        """
        log_ZSMC = tf.reduce_logsumexp(log_Ws, axis=1)  # (time, batch_size)
        log_ZSMC = tf.reduce_sum(tf.reduce_mean(log_ZSMC, axis=1), name="log_ZSMC")

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

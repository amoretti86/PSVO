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

        # bidirectional RNN as full sequence observations encoder
        self.X0_use_separate_RNN = X0_use_separate_RNN
        self.use_stack_rnn = use_stack_rnn

        if model.bRNN is not None:
            self.smooth_obs = True
            self.y_smoother_f, self.y_smoother_b, self.X0_smoother_f, self.X0_smoother_b = model.bRNN
        else:
            self.smooth_obs = False

        self.FFBS = FFBS
        self.FFBS_to_learn = FFBS_to_learn
        self.smoothing_perc = model.smoothing_perc
        if not self.FFBS_to_learn:
            self.smoothing_perc = 1

        self.name = name

    def get_log_ZSMC(self, obs, hidden, Input, q_cov=1, loss_type='main', n_particles=None):
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
            Dx = self.model.Dx
            #n_particles = self.n_particles
            if n_particles is None:
                n_particles = self.n_particles
            batch_size, time, Dy = obs.get_shape().as_list()
            self.batch_size, self.time, self.Dy = batch_size, time, Dy

            # preprossing obs
            X_ancestor, smoothed_obs = self.preprocess_obs(obs)

            # t = 0
            q_f_t_feed = X_ancestor
            if self.use_input:
                q_f_t_feed = tf.concat([q_f_t_feed, Input[:, 0, :]], axis=-1)

            if self.q_uses_true_X:
                X, q_t_log_prob = self.sample_from_true_X(hidden[:, 0, :],
                                                          q_cov,
                                                          sample_shape=n_particles,
                                                          name="q_{}_sample_and_log_prob".format(0))
            else:
                if self.q2 is None:
                    X, q_t_log_prob = self.q0.sample_and_log_prob(q_f_t_feed,
                                                                  sample_shape=n_particles,
                                                                  name="q_{}_sample_and_log_prob".format(0))
                else:
                    X, q_t_log_prob, f_t_log_prob = self.sample_from_2_dist(self.q0,
                                                                            self.q2,
                                                                            q_f_t_feed,
                                                                            smoothed_obs[0],
                                                                            sample_size=n_particles)
            if self.f != self.q1:
                f_t_log_prob = self.f.log_prob(q_f_t_feed, X, name="f_{}_log_prob".format(0))

            g_t_log_prob = self.g.log_prob(X, obs[:, 0], name="g_{}_log_prob".format(0))
            log_W = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_ZSMC".format(0))

            # t = 1, ..., T - 1
            # prepare input tensor arrays
            smoothed_obs_ta = tf.TensorArray(tf.float32, size=time, name="smoothed_obs_ta").unstack(smoothed_obs)
            # prepare state tensor arrays
            log_names = ["X_prevs", "X_ancestors", "log_Ws", "all_fs"]
            log = [tf.TensorArray(tf.float32, size=time, name="{}_ta".format(name)) for name in log_names]
            log_names.append("resample_idx")
            log.append(tf.TensorArray(tf.int32, size=time, name="resample_idx_ta"))

            log[2] = log[2].write(0, log_W)

            def while_cond(t, *unused_args):
                return t < time

            def while_body(t, X_prev, log_W, log):
                X_ancestor, resample_idx = self.resample_X(X_prev, log_W, loss_type=loss_type)
                # resample idx: (n_particles, batch_size)

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
                    if self.q2 is None:
                        X, q_t_log_prob = self.q1.sample_and_log_prob(q_f_t_feed,
                                                                      sample_shape=(),
                                                                      name="q_t_sample_and_log_prob")
                    else:
                        X, q_t_log_prob, f_t_log_prob = self.sample_from_2_dist(self.q1,
                                                                                self.q2,
                                                                                q_f_t_feed,
                                                                                smoothed_obs_ta.read(t),
                                                                                sample_size=())

                if self.FFBS:
                    X_tile = tf.tile(tf.expand_dims(X, axis=1), (1, n_particles, 1, 1), name="X_tile")

                    if self.use_input:
                        Input_t_expanded = tf.tile(tf.expand_dims(Input[:, t, :], axis=0), (n_particles, 1, 1))
                        f_t_all_feed = tf.concat([X_prev, Input_t_expanded], axis=-1)
                    else:
                        f_t_all_feed = X_prev

                    f_t_all_log_prob = self.f.log_prob(f_t_all_feed, X_tile, name="f_t_all_log_prob")

                    f_t_log_prob = self.f.log_prob(q_f_t_feed, X, name="f_t_log_prob")

                else:
                    f_t_all_log_prob = tf.zeros((n_particles, n_particles, batch_size))
                    if self.f != self.q1:
                        f_t_log_prob = self.f.log_prob(q_f_t_feed, X, name="f_t_log_prob")

                g_t_log_prob = self.g.log_prob(X, obs[:, t], name="g_t_log_prob")
                log_W = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_W_t")

                # write to tensor arrays
                #idxs = [t, t, t + 1, t, t]
                idxs = [t-1, t-1, t, t-1, t-1]
                log_contents = [X_prev, X_ancestor, log_W, f_t_all_log_prob, resample_idx]
                log = [ta.write(idx, log_content) for ta, idx, log_content in zip(log, idxs, log_contents)]

                return (t + 1, X, log_W, log)

            # conduct the while loop
            init_state = (1, X, log_W, log)
            t_final, X_T, log_W, log = tf.while_loop(while_cond, while_body, init_state)

            # write to tensor arrays for t = T - 1
            X_T_resampled, resample_idx = self.resample_X(X_T, log_W, loss_type=loss_type)
            log_contents = [X_T, X_T_resampled, log_W, tf.zeros((n_particles, n_particles, batch_size)), resample_idx]
            log = [ta.write(t_final - 1, log_content) if i != 2 else ta
                   for ta, log_content, i in zip(log, log_contents, range(5))]

            # transfer tensor arrays to tensors
            log_shapes = [(time, n_particles, batch_size, Dx)] * 2 + \
                         [(time, n_particles, batch_size), (time, n_particles, n_particles, batch_size),
                          (time, n_particles, batch_size)]
            log = [ta.stack(name=name) for ta, name in zip(log, log_names)]

            for tensor, shape in zip(log, log_shapes):
                tensor.set_shape(shape)

            X_prevs, X_ancestors, log_Ws, all_fs, resample_idxs = log

            # FFBS and computing log_ZSMC
            reweighted_log_Ws = None
            if self.FFBS:
                reweighted_log_Ws = self.reweight_log_Ws(log_Ws, all_fs)
                if self.FFBS_to_learn:
                    log_ZSMC = self.compute_log_ZSMC(reweighted_log_Ws)
                else:
                    log_ZSMC = self.compute_log_ZSMC(log_Ws)

                Xs = [self.resample_X(X, log_W, loss_type=loss_type)
                      for X, log_W, t in zip(tf.unstack(X_prevs), tf.unstack(reweighted_log_Ws), range(time))]
                Xs = tf.stack(Xs)
            else:
                log_ZSMC = self.compute_log_ZSMC(log_Ws)
                Xs = X_ancestors

            # (batch_size, time, n_particles, Dx)
            Xs          = tf.transpose(Xs,          perm=[2, 0, 1, 3], name="Xs")
            X_prevs     = tf.transpose(X_prevs,     perm=[2, 0, 1, 3], name="X_prev")
            X_ancestors = tf.transpose(X_ancestors, perm=[2, 0, 1, 3], name="X_ancestors")

            log = {"Xs":          Xs,
                   "X_prevs":     X_prevs,
                   "X_ancestors": X_ancestors}

            if loss_type == 'full':
                if n_particles == 1:
                    resampling_loss = 0
                else:
                    # calculate the loss w.r.t. resampling step
                    logit_log_Ws = log_Ws[0:time - 1]
                    logit_log_Ws = tf.transpose(logit_log_Ws, perm=[2, 0, 1],
                                                name="logit_log_Ws")  # (batch_size, time-1, n_particles)

                    resample_idxs = resample_idxs[0:time - 1]
                    resample_idxs = tf.transpose(resample_idxs, perm=[1, 2, 0],
                                                 name="resample_idxs")  # (n_particles, batch_size, time-1)

                    def resample_loss_cond(particle_idx, output_ta_1):
                        return tf.less(particle_idx, n_particles)

                    def resample_loss_body(particle_idx, output_ta_1):

                        recover_idx_per_particle = resample_idxs[particle_idx]  # (batch_size, time-1)
                        grad_sampling_per_particle = \
                            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=recover_idx_per_particle, logits=logit_log_Ws)

                        output_ta_1 = output_ta_1.write(particle_idx, grad_sampling_per_particle)

                        return particle_idx + 1, output_ta_1

                    resampling_ta = tf.TensorArray(dtype=tf.float32, size=n_particles)
                    _, final_resampling_ta = tf.while_loop(resample_loss_cond, resample_loss_body, loop_vars=[0, resampling_ta])

                    final_resampling_ta = final_resampling_ta.stack()  # shape (n_particles, batch_size, time-1)

                    resampling_loss = tf.multiply(tf.reduce_sum(final_resampling_ta), tf.stop_gradient(log_ZSMC),
                                                  name="resampling_loss")
            else:
                resampling_loss = 0

        return log_ZSMC, log, resampling_loss

    def sample_from_2_dist(self, dist1, dist2, d1_input, d2_input, sample_size=()):
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

    def sample_from_true_X(self, hidden, q_cov, sample_shape=(), name="q_t_mvn"):
        Dx = self.model.Dx
        mvn = tfd.MultivariateNormalDiag(hidden,
                                         q_cov * tf.ones(Dx, dtype=tf.float32),
                                         name=name)
        X = mvn.sample(sample_shape)
        q_t_log_prob = mvn.log_prob(X)

        return X, q_t_log_prob

    def resample_X(self, X, log_W, loss_type='main'):
        n_particles, batch_size = log_W.get_shape().as_list()

        if loss_type == 'soft':
            return self.soft_sample_X(X, log_W)

        if n_particles > 1:
            resample_idx = self.get_resample_idx(log_W)  # (n_particles, batch_size, 2)
            X_resampled = tf.gather_nd(X, resample_idx)

            # recover resample_idx
            recover_idx = resample_idx[:, :, 0]  # (n_particles, batch_size)
        else:
            X_resampled = X
            recover_idx = tf.constant([[0]])

        return X_resampled, recover_idx

    def soft_sample_X(self, X, log_W):

        n_particles, batch_size = log_W.get_shape().as_list()

        # log_W shape: (n_particles, batch_size)
        log_W_max = tf.stop_gradient(tf.reduce_max(log_W, axis=0))
        log_W = tf.transpose(log_W - log_W_max)  # shape (batch_size, n_particles)

        #soft_categorical = tfd.RelaxedOneHotCategorical(logits=log_W, temperature=0.2)
        soft_categorical = tfd.RelaxedOneHotCategorical(logits=log_W, temperature=1/n_particles)

        soft_idx = soft_categorical.sample(n_particles)  # (n_particles, batch_size, n_particles)

        hard_idx = tf.cast(tf.equal(soft_idx, tf.reduce_max(soft_idx, 2, keepdims=True)), soft_idx.dtype)

        idx = tf.stop_gradient(hard_idx - soft_idx) + soft_idx

        idx = tf.transpose(idx, perm=[1, 0, 2])  # (batch_size, n_particles, n_particles)

        # X shape: (n_particles, batch_size, Dx)
        X = tf.transpose(X, perm=[1, 0, 2])  # (batch_size, n_particles, Dx)
        X_resampled = tf.matmul(idx, X)  # (batch_size, n_particles, Dx)
        X_resampled = tf.transpose(X_resampled, perm=[1, 0, 2])

        recover_idx = tf.zeros([n_particles, batch_size])  # would not be used
        return X_resampled, recover_idx

    def get_resample_idx(self, log_W, ):
        # get resample index a_t^k ~ Categorical(w_t^1, ..., w_t^K)
        # (n_particles, batch_size)

        n_particles, batch_size = log_W.get_shape().as_list()

        log_W_max = tf.stop_gradient(tf.reduce_max(log_W, axis=0))
        log_W = tf.transpose(log_W - log_W_max)  # shape (batch_size, n_particles)

        categorical = tfd.Categorical(logits=log_W, validate_args=True,
                                      name="Categorical_t")

        # sample multiple times to remove idx out of range
        idx = tf.ones((n_particles, batch_size), dtype=tf.int32) * n_particles
        for _ in range(3):
            fixup_idx = categorical.sample(n_particles)
            idx = tf.where(idx >= n_particles, fixup_idx, idx)

        # if still got idx out of range, replace them with idx from uniform distribution
        final_fixup = tf.random.uniform((n_particles, batch_size),
                                        maxval=n_particles, dtype=tf.int32)
        idx = tf.where(idx >= n_particles, final_fixup, idx)

        # ugly stuff used to resample X
        batch_1xB = tf.expand_dims(tf.range(batch_size), axis=0)  # (1, batch_size)
        batch_NxB = tf.tile(batch_1xB, (n_particles, 1))  # (n_particles, batch_size)

        resample_idx_NxBx2 = tf.stack((idx, batch_NxB), axis=2)  # (n_particles, batch_size, 2)

        return resample_idx_NxBx2

    def reweight_log_Ws(self, log_Ws, all_fs):
        # reweight weight of each particle (w_t^k) according to FFBS formula
        log_Ws, all_fs = tf.unstack(log_Ws), tf.unstack(all_fs)
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

    def preprocess_obs(self, obs):
        # if self.smooth_obs, smooth obs with bidirectional RNN

        if not self.smooth_obs:
            preprocessed_obs = tf.unstack(obs, axis=1)
            preprocessed_X0 = preprocessed_obs[0]

            return preprocessed_X0, preprocessed_obs

        with tf.variable_scope("smooth_obs"):

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

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from SMC.SMC_base import SMC


class FFBSimulationMix(SMC):
    def __init__(self, model, FLAGS, name="log_ZSMC"):
        SMC.__init__(self, model, FLAGS, name="log_ZSMC")
        self.BSim_sample_new_particles = FLAGS.BSim_sample_new_particles
        if self.BSim_sample_new_particles:

            self.n_particles_for_BSim_proposal = FLAGS.n_particles_for_BSim_proposal

            self.q1_inv = model.q1_inv_dist
            self.smooth_obs = self.FF_use_bRNN = FLAGS.FF_use_bRNN
            self.BSim_use_single_RNN = FLAGS.BSim_use_single_RNN
            if self.FF_use_bRNN and not self.BSim_use_single_RNN:
                self.BSim_q2 = model.q2_dist
            else:
                self.BSim_q2 = model.BSim_q2_dist

    def get_log_ZSMC(self, obs, hidden):
        """
        Get log_ZSMC from obs y_1:T
        Input:
            obs.shape = (batch_size, time, Dy)
            hidden.shape = (batch_size, time, Dz)
        Output:
            log_ZSMC: shape = scalar
            log: stuff to debug
        """
        batch_size, time, _ = obs.get_shape().as_list()
        self.Dx, self.batch_size, self.time = self.model.Dx, batch_size, time

        with tf.variable_scope(self.name):

            log = {}

            # get X_1:T, resampled X_1:T and log(W_1:T) from SMC
            X_prevs, _, log_Ws = self.SMC(hidden, obs)
            if self.BSim_sample_new_particles:
                bw_Xs, FFBS_log_weights = self.backward_simulation_w_proposal(X_prevs, log_Ws, obs)

            log_ZSMC = self.compute_log_ZSMC(FFBS_log_weights)
            Xs = bw_Xs

            # shape = (batch_size, time, n_particles, Dx)
            Xs = tf.transpose(Xs, perm=[2, 0, 1, 3], name="Xs")

            log["Xs"] = Xs

        return log_ZSMC, log

    @staticmethod
    def compute_log_ZSMC(log_Ws):
        """
        :param log_Ws: shape (time, n_particles, batch_size)
        :return: loss, shape ()
        """
        log_ZSMC = tf.reduce_logsumexp(log_Ws, axis=1)  # (time, batch_size)
        log_ZSMC = tf.reduce_sum(tf.reduce_mean(log_ZSMC, axis=1), name="log_ZSMC")

        return log_ZSMC

    def backward_simulation_w_proposal(self, Xs, log_Ws, obs):
        """

        :param Xs: shape (time, n_particles, batch_size, Dx)
        :param log_Ws: shape (time, n_particles, batch_size)
        :param obs: shpae (batch_size, time, Dy)
        :return:
        """
        Dx, time, n_particles, batch_size = self.Dx, self.time, self.n_particles, self.batch_size
        M = self.n_particles_for_BSim_proposal

        # store in reverse order
        bw_Xs_ta = tf.TensorArray(tf.float32, size=time, name="backward_X_ta")
        FFBS_log_weights_ta = tf.TensorArray(tf.float32, size=time, name="FFBS_log_weights_ta")

        # t = T - 1
        # sample M particles
        X_Tm1, log_Tm1 = Xs[-1], log_Ws[-1] - tf.reduce_logsumexp(log_Ws[-1], axis=0)
        bw_X_Tm1, bw_log_W_Tm1 = self.resample_X([X_Tm1, log_Tm1], log_Tm1, sample_size=n_particles)
        g_Tm1_log_prob = self.g.log_prob(bw_X_Tm1, obs[:, time - 1])

        bw_Xs_ta = bw_Xs_ta.write(time - 1, bw_X_Tm1)
        preprocessed_obs = self.BS_preprocess_obs(obs)
        preprocessed_obs_ta = \
            tf.TensorArray(tf.float32, size=time, name="preprocessed_obs_ta").unstack(preprocessed_obs)

        #  from t = T - 2 to 1
        def while_cond(t, *unused_args):
            return t >= 1

        def while_body(t, bw_X_tp1, bw_log_W_tp1, bw_Xs_ta, FFBS_log_weights_ta):
            # proposal q(x_t | x_t+1, y_t)
            # bw_X_t.shape = (M, n_particles, batch_size, Dx)
            # bw_q_log_prob.shape = (M, n_particles, batch_size)
            bw_X_t, bw_q_log_prob, _ = self.sample_from_2_dist(self.q1_inv, self.BSim_q2,
                                                               bw_X_tp1, preprocessed_obs_ta.read(t),
                                                               sample_size=M)

            # f(x_t+1 | x_t) (M, n_particles, batch_size)
            f_t_log_prob = self.f.log_prob(bw_X_t, bw_X_tp1)

            # p(x_t | y_{1:t}) is proprotional to \int p(x_t-1 | y_{1:t-1}) * f(x_t | x_t-1) dx_t-1 * g(y_t | x_t)
            bw_X_t_tiled = tf.tile(tf.expand_dims(bw_X_t, axis=2), (1, 1, n_particles, 1, 1))
            f_tm1_log_prob = self.f.log_prob(Xs[t - 1], bw_X_t_tiled)   # (M, n_particles, n_particles, batch_size)
            g_t_log_prob = self.g.log_prob(bw_X_t, obs[:, t])           # (M, n_particles, batch_size)

            # p (x_t | y_{1:t})
            log_W_tm1 = log_Ws[t - 1] - tf.reduce_logsumexp(log_Ws[t - 1], axis=0)
            log_W_t = tf.reduce_logsumexp(f_tm1_log_prob + log_W_tm1, axis=2) + g_t_log_prob
            assert log_W_t.shape.as_list() == [M, n_particles, batch_size]

            # p(x_t | x_{t+1:T}, y_{1:T})
            bw_log_W_t = log_W_t + f_t_log_prob - bw_q_log_prob
            bw_log_W_t = bw_log_W_t - tf.reduce_logsumexp(bw_log_W_t, axis=0)
            assert bw_log_W_t.shape.as_list() == [M, n_particles, batch_size]

            bw_X_t, bw_log_W_t, f_t_log_prob, g_t_log_prob, selected_fw_log_normalized_W_t = \
                self.resample_X([bw_X_t, bw_log_W_t, f_t_log_prob, g_t_log_prob, log_W_t],
                                bw_log_W_t, sample_size=())

            assert bw_X_t.shape.as_list() == [n_particles, batch_size, Dx]

            selected_g_log_prob_tp1 = self.g.log_prob(bw_X_tp1, obs[:, t+1])
            assert selected_g_log_prob_tp1.shape.as_list() == [n_particles, batch_size]

            FFBS_log_weights_t = tf.add(selected_fw_log_normalized_W_t + f_t_log_prob + selected_g_log_prob_tp1 \
                                        - bw_log_W_tp1, -bw_log_W_t, name="FFBS_log_weights_t")

            bw_Xs_ta = bw_Xs_ta.write(t, bw_X_t)
            FFBS_log_weights_ta = FFBS_log_weights_ta.write(t+1, FFBS_log_weights_t)

            return t - 1, bw_X_t, bw_log_W_t, bw_Xs_ta, FFBS_log_weights_ta

        # conduct the while loop
        init_state = (time - 2, bw_X_Tm1, bw_log_W_Tm1, bw_Xs_ta, FFBS_log_weights_ta)
        t, bw_X_1, bw_log_W_1, bw_Xs_ta, FFBS_log_weights_ta = \
            tf.while_loop(while_cond, while_body, init_state)

        # t = 0
        # bw_X_t.shape = (M, n_particles, batch_size, Dx)
        # bw_q_log_prob.shape = (M, n_particles, batch_size)
        bw_X_0, bw_q_log_prob, _ = self.sample_from_2_dist(self.q1_inv, self.BSim_q2,
                                                           bw_X_1, preprocessed_obs[0],
                                                           sample_size=M)

        f_0_log_prob = self.f.log_prob(bw_X_0, bw_X_1)          # (M, n_particles, batch_size)
        g_0_log_prob = self.g.log_prob(bw_X_0, obs[:, 0])       # (M, n_particles, batch_size)

        # self.preprocessed_X0_f is from self.SMC()
        mu_0 = self.preprocessed_X0
        if not (self.model.use_bootstrap and self.model.use_2_q):
            f_init_log_prob = self.f.log_prob(mu_0, bw_X_0)     # (M, n_particles, batch_size)
        else:
            f_init_log_prob = self.q0.log_prob(mu_0, bw_X_0)    # (M, n_particles, batch_size)

        log_W_0 = f_init_log_prob + g_0_log_prob

        bw_log_W_0 = log_W_0 + f_0_log_prob - bw_q_log_prob
        bw_log_W_0 = bw_log_W_0 - tf.reduce_logsumexp(bw_log_W_0, axis=0)
        bw_X_0, bw_log_W_0, f_0_log_prob, f_init_log_prob, g_0_log_prob, selected_fw_log_normalized_W_0 = \
            self.resample_X([bw_X_0, bw_log_W_0, f_0_log_prob, f_init_log_prob, g_0_log_prob, log_W_0],
                            bw_log_W_0,
                            sample_size=())

        selected_g_log_prob_1 = self.g.log_prob(bw_X_1, obs[:, 1])
        FFBS_log_weights_1 = tf.add(selected_fw_log_normalized_W_0 + f_0_log_prob + selected_g_log_prob_1 \
                                    - bw_log_W_1, -bw_log_W_0, name="FFBS_log_weights_1")

        bw_Xs_ta = bw_Xs_ta.write(0, bw_X_0)
        FFBS_log_weights_ta = FFBS_log_weights_ta.write(1, FFBS_log_weights_1)

        # t=0 for FFBS_log_weights
        selected_f_log_prob_0 = self.q0.log_prob(mu_0, bw_X_0, name="selected_f_log_prob_0")
        selected_g_log_prob_0 = self.g.log_prob(bw_X_0, obs[:, 0], name="selected_g_lor_prob_0")

        FFBS_log_weights_0 = tf.add(selected_f_log_prob_0 + selected_g_log_prob_0, - bw_log_W_0,
                                    name="FFBS_log_weights_0")
        FFBS_log_weights_ta = FFBS_log_weights_ta.write(0, FFBS_log_weights_0)


        # transfer tensor arrays to tensors
        bw_Xs = bw_Xs_ta.stack()
        FFBS_log_weights = FFBS_log_weights_ta.stack()

        bw_Xs.set_shape((time, n_particles, batch_size, Dx))
        FFBS_log_weights.set_shape((time, n_particles, batch_size))

        return bw_Xs, FFBS_log_weights

    def BS_preprocess_obs(self, obs):
        # if self.smooth_obs, smooth obs with bidirectional RNN
        if self.FF_use_bRNN and not self.BSim_use_single_RNN:
            return self.preprocessed_obs

        with tf.variable_scope("smooth_obs"):
            if self.BSim_use_single_RNN:
                cells = self.y_smoother_f
                if isinstance(self.y_smoother_f, list):
                    cells = tf.nn.rnn_cell.MultiRNNCell(self.y_smoother_f)
                preprocessed_obs, _ = tf.nn.static_rnn(cells, tf.unstack(obs, axis=1), dtype=tf.float32)
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

        return preprocessed_obs

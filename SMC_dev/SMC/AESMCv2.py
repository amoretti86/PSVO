import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from SMC.SMC_base import SMC


class AESMCv2(SMC):
    def __init__(self, model, FLAGS, name="log_ZSMC"):
        SMC.__init__(self, model, FLAGS, name="log_ZSMC")

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
            X_prevs, X_ancestors, q_t_log_probs, f_t_log_probs, g_t_log_probs = self.SMC(hidden, obs)

            log_ZSMC = self.compute_log_ZSMC(q_t_log_probs, f_t_log_probs, g_t_log_probs)
            Xs = X_ancestors

            # shape = (batch_size, time, n_particles, Dx)
            Xs = tf.transpose(Xs, perm=[2, 0, 1, 3], name="Xs")

            log["Xs"] = Xs

        return log_ZSMC, log

    @staticmethod
    def compute_log_ZSMC(q_t_log_probs, f_t_log_probs, g_t_log_probs):
        """
        Input:
            q_t_log_probs.shape = (time, n_particles, batch_size)
            f_t_log_probs.shape = (time, n_particles, batch_size)
            g_t_log_probs.shape = (time, n_particles, batch_size)
        """
        time, n_particles, batch_size = q_t_log_probs.get_shape().as_list()

        joint = tf.reduce_sum(f_t_log_probs + g_t_log_probs, axis=0)
        proposal = tf.reduce_sum(q_t_log_probs, axis=0)
        log_ZSMC = tf.reduce_logsumexp(joint - proposal, axis=0) - tf.log(tf.constant(n_particles, dtype=tf.float32))
        log_ZSMC = tf.reduce_mean(log_ZSMC)

        return log_ZSMC

    def SMC(self, hidden, obs, q_cov=1.0):
        Dx, time, n_particles, batch_size = self.Dx, self.time, self.n_particles, self.batch_size

        # preprossing obs
        preprocessed_X0, preprocessed_obs = self.preprocess_obs(obs)
        self.preprocessed_X0  = preprocessed_X0
        self.preprocessed_obs = preprocessed_obs
        q0, q1, f = self.q0, self.q1, self.f

        # -------------------------------------- t = 0 -------------------------------------- #
        q_f_0_feed = preprocessed_X0

        # proposal
        if self.q_uses_true_X:
            X, q_0_log_prob = self.sample_from_true_X(hidden[:, 0, :],
                                                      q_cov,
                                                      sample_shape=n_particles,
                                                      name="q_{}_sample_and_log_prob".format(0))
        else:
            if self.model.use_2_q:
                X, q_0_log_prob, f_0_log_prob = self.sample_from_2_dist(q0,
                                                                        self.q2,
                                                                        q_f_0_feed,
                                                                        preprocessed_obs[0],
                                                                        sample_size=n_particles)
            else:
                X, q_0_log_prob = q0.sample_and_log_prob(q_f_0_feed,
                                                         sample_shape=n_particles,
                                                         name="q_{}_sample_and_log_prob".format(0))
        # transition log probability
        # only when use_bootstrap and use_2_q, f_t_log_prob has been calculated
        if not (self.model.use_bootstrap and self.model.use_2_q):
            f_0_log_prob = f.log_prob(q_f_0_feed, X, name="f_{}_log_prob".format(0))

        # emission log probability and log weights
        g_0_log_prob = self.g.log_prob(X, obs[:, 0], name="g_{}_log_prob".format(0))

        log_W = tf.add(f_0_log_prob, g_0_log_prob - q_0_log_prob, name="log_W_{}".format(0))

        # -------------------------------------- t = 1, ..., T - 1 -------------------------------------- #
        # prepare tensor arrays
        # tensor arrays to read
        preprocessed_obs_ta = \
            tf.TensorArray(tf.float32, size=time, name="preprocessed_obs_ta").unstack(preprocessed_obs)

        # tensor arrays to write
        # particles, resampled particles (mean), log weights of particles
        log_names = ["X_prevs", "X_ancestors", "q_t_log_probs", "f_log_probs", "g_log_probs"]
        log = [tf.TensorArray(tf.float32, size=time, clear_after_read=False, name="{}_ta".format(name))
               for name in log_names]

        # write results for t = 0 into tensor arrays
        log[2] = log[2].write(0, q_0_log_prob)
        log[3] = log[3].write(0, f_0_log_prob)
        log[4] = log[4].write(0, g_0_log_prob)

        def while_cond(t, *unused_args):
            return t < time

        def while_body(t, X_prev, log_W, log):
            # resampling
            X_ancestor = self.resample_X(X_prev, log_W, sample_size=n_particles)

            q_f_t_feed = X_ancestor

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
            log_W = tf.add(f_t_log_prob, g_t_log_prob - q_t_log_prob, name="log_W_t")

            # write results in this loop to tensor arrays
            idxs = [t - 1, t - 1, t, t, t]
            log_contents = [X_prev, X_ancestor, q_t_log_prob, f_t_log_prob, g_t_log_prob]
            log = [ta.write(idx, log_content) for ta, idx, log_content in zip(log, idxs, log_contents)]

            return (t + 1, X, log_W, log)

        # conduct the while loop
        init_state = (1, X, log_W, log)
        t_final, X_T, log_W, log = tf.while_loop(while_cond, while_body, init_state)

        # write final results at t = T - 1 to tensor arrays
        X_T_resampled = self.resample_X(X_T, log_W, sample_size=n_particles)
        log[0] = log[0].write(t_final - 1, X_T)
        log[1] = log[1].write(t_final - 1, X_T_resampled)

        # convert tensor arrays to tensors
        log_shapes = [(time, n_particles, batch_size, Dx)] * 2 + [(time, n_particles, batch_size)] * 3
        log = [ta.stack(name=name) for ta, name in zip(log, log_names)]
        for tensor, shape in zip(log, log_shapes):
            tensor.set_shape(shape)

        X_prevs, X_ancestors, q_t_log_probs, f_t_log_probs, g_t_log_probs = log

        return X_prevs, X_ancestors, q_t_log_probs, f_t_log_probs, g_t_log_probs

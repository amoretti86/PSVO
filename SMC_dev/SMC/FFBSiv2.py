import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from SMC.SMC_base import SMC


class FFBSiv2(SMC):
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

            self.BSim_q_init = model.Bsim_q_init_dist

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

            Xs, q_log_probs, f_log_probs, g_log_probs = self.backward_simulation_w_proposal(obs)

            log_ZSMC = self.compute_log_ZSMC(q_log_probs, f_log_probs, g_log_probs)

            # shape = (batch_size, time, n_particles, Dx)
            Xs = tf.transpose(Xs, perm=[2, 0, 1, 3], name="Xs")

            log["Xs"] = Xs

        return log_ZSMC, log

    @staticmethod
    def compute_log_ZSMC(bw_log_Ws, f_log_probs, g_log_probs):
        """
        Input:
            bw_log_Ws.shape = (time, n_particles, batch_size)
            f_log_probs.shape = (time, n_particles, batch_size)
            g_log_probs.shape = (time, n_particles, batch_size)
        """
        time, n_particles, batch_size = bw_log_Ws.get_shape().as_list()

        joint = tf.reduce_sum(f_log_probs + g_log_probs, axis=0)
        proposal = tf.reduce_sum(bw_log_Ws, axis=0)
        log_ZSMC = tf.reduce_logsumexp(joint - proposal, axis=0) - tf.log(tf.constant(n_particles, dtype=tf.float32))
        log_ZSMC = tf.reduce_mean(log_ZSMC)

        return log_ZSMC

    def backward_simulation_w_proposal(self, obs):
        Dx, time, n_particles, batch_size = self.Dx, self.time, self.n_particles, self.batch_size

        # store in reverse order
        Xs_ta = tf.TensorArray(tf.float32, size=time, name="backward_X_ta")
        q_log_probs_ta = tf.TensorArray(tf.float32, size=time, name="q_log_probs")
        f_log_probs_ta = tf.TensorArray(tf.float32, size=time, name="joint_f_log_probs")
        g_log_probs_ta = tf.TensorArray(tf.float32, size=time, name="joint_g_log_probs")

        self.preprocessed_X0, self.preprocessed_obs = self.preprocess_obs(obs)

        # t = T - 1
        # proposal q(x_T | y_{1:T})
        X_Tm1, q_Tm1_log_prob = \
            self.BSim_q_init.sample_and_log_prob(self.preprocessed_obs[-1], sample_shape=n_particles)

        g_Tm1_log_prob = self.g.log_prob(X_Tm1, obs[:, time-1])  # (n_particles, batch_size)

        Xs_ta = Xs_ta.write(time - 1, X_Tm1)
        q_log_probs_ta = q_log_probs_ta.write(time - 1, q_Tm1_log_prob)
        g_log_probs_ta = g_log_probs_ta.write(time - 1, g_Tm1_log_prob)
        preprocessed_obs_ta = \
            tf.TensorArray(tf.float32, size=time, name="preprocessed_obs_ta").unstack(self.preprocessed_obs)

        #  from t = T - 2 to 1
        def while_cond(t, *unused_args):
            return t >= 0

        def while_body(t, X_tp1, Xs_ta, q_log_probs_ta, f_log_probs_ta, g_log_probs_ta):
            # proposal q(x_t | x_t+1, y_{1:T})
            # X_t.shape = (n_particles, batch_size, Dx)
            # q_log_prob.shape = (n_particles, batch_size)
            X_t, q_t_log_prob, _ = self.sample_from_2_dist(self.q1_inv, self.BSim_q2,
                                                               X_tp1, preprocessed_obs_ta.read(t))

            # f(x_t+1 | x_t) (M, n_particles, batch_size)
            f_t_log_prob = self.f.log_prob(X_t, X_tp1)

            g_t_log_prob = self.g.log_prob(X_t, obs[:, t])           # (n_particles, batch_size)

            Xs_ta = Xs_ta.write(t, X_t)
            q_log_probs_ta = q_log_probs_ta.write(t, q_t_log_prob)
            f_log_probs_ta = f_log_probs_ta.write(t + 1, f_t_log_prob)
            g_log_probs_ta = g_log_probs_ta.write(t, g_t_log_prob)

            return t - 1, X_t, Xs_ta, q_log_probs_ta, f_log_probs_ta, g_log_probs_ta

        # conduct the while loop
        init_state = (time - 2, X_Tm1, Xs_ta, q_log_probs_ta, f_log_probs_ta, g_log_probs_ta)
        t, X_0, Xs_ta, q_log_probs_ta, f_log_probs_ta, g_log_probs_ta = \
            tf.while_loop(while_cond, while_body, init_state)

        # t = 0
        # self.preprocessed_X0_f is from self.SMC()
        mu_0 = self.preprocessed_X0
        if not (self.model.use_bootstrap and self.model.use_2_q):
            f_init_log_prob = self.f.log_prob(mu_0, X_0)     # (n_particles, batch_size)
        else:
            f_init_log_prob = self.q0.log_prob(mu_0, X_0)    # (n_particles, batch_size)

        f_log_probs_ta = f_log_probs_ta.write(0, f_init_log_prob)

        # transfer tensor arrays to tensors
        Xs = Xs_ta.stack()
        q_log_probs = q_log_probs_ta.stack()
        f_log_probs = f_log_probs_ta.stack()
        g_log_probs = g_log_probs_ta.stack()

        Xs.set_shape((time, n_particles, batch_size, Dx))
        q_log_probs.set_shape((time, n_particles, batch_size))
        f_log_probs.set_shape((time, n_particles, batch_size))
        g_log_probs.set_shape((time, n_particles, batch_size))

        return Xs, q_log_probs, f_log_probs, g_log_probs


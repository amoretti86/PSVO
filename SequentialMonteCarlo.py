import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import pdb

class SequentialMonteCarlo:

    def __init__(self, obs, n_particles, x_0, q, f, g, p, name="SMC_init"):
        with tf.name_scope(name):
            #pdb.set_trace()
            self.Dx = x_0.get_shape().as_list()[0]
            self.n_particles = n_particles
            self.T, Dy = obs.get_shape().as_list()#[0] (Time, Dy) if not batch
            self.x_0 = x_0
            self.q = q
            self.f = f
            self.g = g
            self.p = p

    def get_log_ZSMC(self, obs, use_stop_gradient=True, name="get_log_ZSMC"):
        """
        Run SMC symbolically in TensorFlow to compute filtered paths,
        ancestor indicies and weights. The log weights are used to construct
        a loss function as an estimate for the log likelihood.
        """
        print("Here is the g function:", self.g)
        with tf.name_scope(name):

            #Dx = x_0.get_shape().as_list()[0]
            #T, Dy = obs.get_shape().as_list()
            # Ugly to use lists. Replace with Tensors

            Xs = []
            Ws = []
            W_means = []
            fs = []
            gs = []
            qs = []
            ps = []
            #pdb.set_trace()
            # T = 1
            X = self.q.sample(None, name='X0')
            print("X.shape", X.shape)
            q_uno_probs = self.q.prob(None, X, name='q_uno_probs')
            f_nu_probs = self.f.prob(None, X, name='f_nu_probs')
            g_uno_probs = self.g.prob(X, obs[0], name='g_uno_probs')

            W = tf.multiply(g_uno_probs, f_nu_probs / q_uno_probs, name='W_0')
            self.log_ZSMC = tf.log(tf.reduce_mean(W, name='W_0_mean'), name='log_ZSMC_0')

            Xs.append(X)
            Ws.append(W)
            W_means.append(tf.reduce_mean(W))
            fs.append(f_nu_probs)
            gs.append(g_uno_probs)
            qs.append(q_uno_probs)
            ps.append(tf.zeros(self.n_particles))

            for t in range(1, self.T):

                # W_{t-1} = W_{t-1} * p(y_t | X_{t-1})
                # k = p.posterior(X, obs[t], name = 'p_{}'.format(t))
                k = tf.ones(self.n_particles, dtype=tf.float32)
                W = W * k

                categorical = tfd.Categorical(probs=W / tf.reduce_sum(W), name='Categorical_{}'.format(t))
                if use_stop_gradient:
                    idx = tf.stop_gradient(categorical.sample(self.n_particles))
                else:
                    idx = categorical.sample(self.n_particles)

                X_prev = tf.gather(X, idx, validate_indices=True)

                X = self.q.sample(X_prev, name='q_{}_sample'.format(t))
                q_t_probs = self.q.prob(X_prev, X, name='q_{}_probs'.format(t))
                f_t_probs = self.f.prob(X_prev, X, name='f_{}_probs'.format(t))
                g_t_probs = self.g.prob(X, obs[t], name='g_{}_probs'.format(t))

                W = tf.divide(g_t_probs * f_t_probs, k * q_t_probs, name='W_{}'.format(t))
                self.log_ZSMC += tf.log(tf.reduce_mean(W), name='log_ZSMC_{}'.format(t))

                Xs.append(X)
                Ws.append(W)
                W_means.append(tf.reduce_mean(W))
                fs.append(f_t_probs)
                gs.append(g_t_probs)
                qs.append(q_t_probs)
                ps.append(k)

            return self.log_ZSMC, [Xs, Ws, W_means, fs, gs, qs, ps]


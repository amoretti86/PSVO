import tensorflow as tf
from transformation.MLP import MLP_transformation
from distribution.mvn import tf_mvn
from distribution.poisson import tf_poisson


class SSM(object):
    """
    state space model
    keeps all placeholders
    keeps q, f, g and obs smoother
    """

    def __init__(self, FLAGS,
                 use_residual=False,
                 output_cov=False,
                 diag_cov=False,
                 use_bootstrap=True,
                 use_2_q=True,
                 poisson_emission=False,
                 smooth_obs=True,
                 X0_use_separate_RNN=True,
                 use_stack_rnn=False):
        # leave the bool flags so that one can know which flags affect which classes

        self.Dx = FLAGS.Dx
        self.Dy = FLAGS.Dy
        self.Di = FLAGS.Di

        self.time = FLAGS.time
        self.batch_size = FLAGS.batch_size

        # Feed-Forward Network (FFN) architectures
        self.q0_layers = [int(x) for x in FLAGS.q0_layers.split(",")]
        self.q1_layers = [int(x) for x in FLAGS.q1_layers.split(",")]
        self.q2_layers = [int(x) for x in FLAGS.q2_layers.split(",")]
        self.f_layers  = [int(x) for x in FLAGS.f_layers.split(",")]
        self.g_layers  = [int(x) for x in FLAGS.g_layers.split(",")]

        self.q0_sigma_init, self.q0_sigma_min = FLAGS.q0_sigma_init, FLAGS.q0_sigma_min
        self.q1_sigma_init, self.q1_sigma_min = FLAGS.q1_sigma_init, FLAGS.q1_sigma_min
        self.q2_sigma_init, self.q2_sigma_min = FLAGS.q2_sigma_init, FLAGS.q2_sigma_min
        self.f_sigma_init,  self.f_sigma_min  = FLAGS.f_sigma_init, FLAGS.f_sigma_min
        self.g_sigma_init,  self.g_sigma_min  = FLAGS.f_sigma_init, FLAGS.g_sigma_min

        # bidirectional RNN
        self.y_smoother_Dhs  = [int(x) for x in FLAGS.y_smoother_Dhs.split(",")]
        self.X0_smoother_Dhs = [int(x) for x in FLAGS.X0_smoother_Dhs.split(",")]

        self.use_residual = use_residual
        self.output_cov = output_cov
        self.diag_cov = diag_cov
        self.use_bootstrap = use_bootstrap
        self.use_2_q = use_2_q
        self.poisson_emission = poisson_emission
        self.smooth_obs = smooth_obs
        self.X0_use_separate_RNN = X0_use_separate_RNN
        self.use_stack_rnn = use_stack_rnn

        self.init_placeholder()
        self.init_trans()
        self.init_dist()
        self.init_smoother()

    def init_placeholder(self):
        self.obs = tf.placeholder(tf.float32, shape=(self.batch_size, self.time, self.Dy), name="obs")
        self.hidden = tf.placeholder(tf.float32, shape=(self.batch_size, self.time, self.Dx), name="hidden")
        self.Input = tf.placeholder(tf.float32, shape=(self.batch_size, self.time, self.Di), name="Input")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.smoothing_perc = tf.placeholder(tf.float32, name="smoothing_perc")

    def init_trans(self):
        self.q0_tran = MLP_transformation(self.q0_layers, self.Dx,
                                          use_residual=False,
                                          output_cov=self.output_cov,
                                          diag_cov=self.diag_cov,
                                          dropout_rate=self.dropout,
                                          name="q0_tran")
        self.q1_tran = MLP_transformation(self.q1_layers, self.Dx,
                                          use_residual=self.use_residual,
                                          output_cov=self.output_cov,
                                          diag_cov=self.diag_cov,
                                          dropout_rate=self.dropout,
                                          name="q1_tran")
        if self.use_2_q:
            self.q2_tran = MLP_transformation(self.q2_layers, self.Dx,
                                              use_residual=False,
                                              output_cov=self.output_cov,
                                              diag_cov=self.diag_cov,
                                              dropout_rate=self.dropout,
                                              name="q2_tran")
        else:
            self.q2_tran = None

        if self.use_bootstrap:
            self.f_tran = self.q1_tran
        else:
            self.f_tran = MLP_transformation(self.f_layers, self.Dx,
                                             use_residual=self.use_residual,
                                             output_cov=self.output_cov,
                                             diag_cov=self.diag_cov,
                                             dropout_rate=self.dropout,
                                             name="f_tran")

        self.g_tran = MLP_transformation(self.g_layers, self.Dy,
                                         use_residual=False,
                                         output_cov=self.output_cov,
                                         diag_cov=self.diag_cov,
                                         dropout_rate=self.dropout,
                                         name="g_tran")

    def init_dist(self):
        self.q0_dist = tf_mvn(self.q0_tran,
                              sigma_init=self.q0_sigma_init,
                              sigma_min=self.q0_sigma_min,
                              name="q0_dist")
        self.q1_dist = tf_mvn(self.q1_tran,
                              sigma_init=self.q1_sigma_init,
                              sigma_min=self.q1_sigma_min,
                              name="q1_dist")

        if self.use_2_q:
            self.q2_dist = tf_mvn(self.q2_tran,
                                  sigma_init=self.q2_sigma_init,
                                  sigma_min=self.q2_sigma_min,
                                  name="q2_dist")
        else:
            self.q2_dist = None

        if self.use_bootstrap:
            self.f_dist = self.q1_dist
        else:
            self.f_dist = tf_mvn(self.f_tran,
                                 sigma_init=self.f_sigma_init,
                                 sigma_min=self.f_sigma_min,
                                 name="f_dist")
        if self.poisson_emission:
            self.g_dist = tf_poisson(self.g_tran,
                                     name="g_dist")
        else:
            self.g_dist = tf_mvn(self.g_tran,
                                 sigma_init=self.g_sigma_init,
                                 sigma_min=self.g_sigma_min,
                                 name="g_dist")

    def init_smoother(self):
        if self.smooth_obs:
            y_smoother_f = [tf.contrib.rnn.LSTMBlockCell(Dh, name="y_smoother_f_{}".format(i))
                            for i, Dh in enumerate(self.y_smoother_Dhs)]
            y_smoother_b = [tf.contrib.rnn.LSTMBlockCell(Dh, name="y_smoother_b_{}".format(i))
                            for i, Dh in enumerate(self.y_smoother_Dhs)]
            if not self.use_stack_rnn:
                y_smoother_f = tf.nn.rnn_cell.MultiRNNCell(y_smoother_f)
                y_smoother_b = tf.nn.rnn_cell.MultiRNNCell(y_smoother_b)

            if self.X0_use_separate_RNN:
                X0_smoother_f = [tf.contrib.rnn.LSTMBlockCell(Dh, name="X0_smoother_f_{}".format(i))
                                 for i, Dh in enumerate(self.X0_smoother_Dhs)]
                X0_smoother_b = [tf.contrib.rnn.LSTMBlockCell(Dh, name="X0_smoother_b_{}".format(i))
                                 for i, Dh in enumerate(self.X0_smoother_Dhs)]
                if not self.use_stack_rnn:
                    X0_smoother_f = tf.nn.rnn_cell.MultiRNNCell(X0_smoother_f)
                    X0_smoother_b = tf.nn.rnn_cell.MultiRNNCell(X0_smoother_b)
            else:
                X0_smoother_f = X0_smoother_b = None

            self.bRNN = (y_smoother_f, y_smoother_b, X0_smoother_f, X0_smoother_b)

        else:
            self.bRNN = None

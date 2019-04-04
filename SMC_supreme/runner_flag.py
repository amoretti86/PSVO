import tensorflow as tf
import tensorflow_probability as tfp
import os
import numpy as np

from SMC_supreme.runner import main 

np.warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # to avoid lots of log about the device

print("the code is written in:")
print("\t tensorflow version: 1.12.0")
print("\t tensorflow_probability version: 0.5.0")

print("the system uses:")
print("\t tensorflow version:", tf.__version__)
print("\t tensorflow_probability version:", tfp.__version__)


# --------------------- training hyperparameters --------------------- #
Dx = 2
Dy = 1
Di = 1
n_particles = 32

batch_size = 1
lr = 2e-4
epoch = 300
seed = 0

# ------------------- data set parameters ------------------ #
# generate synthetic data?
generateTrainingData = False

# if reading data from file
datadir = "/Users/leah/Columbia/courses/19Spring/research/VISMC/data/fhn/[1,0]_obs_cov_0.01/"
# "/Users/leah/Columbia/courses/19Spring/research/VISMC/data/fhn/[1,0]_obs_cov_0.01/"
# "C:/Users/admin/Desktop/research/VISMC/code/VISMC/data/fhn/[1,0]_obs_cov_0.01/"
# "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/lorenz/[1,0,0]_obs_cov_0.4/"
datadict = "datadict"
isPython2 = True

# time, n_train and n_test will be overwritten if loading data from the file
time = 200
n_train = 200 * batch_size
n_test = 40 * batch_size

# -------------------- model parameters -------------------- #
# Feed-Forward Network (FFN)
q0_layers = [64]        # q(x_1|y_1) or q(x_1|y_1:T)
q1_layers = [64]        # q(x_t|x_{t-1})
q2_layers = [64]        # q(x_t|y_t) or q(x_t|y_1:T)
f_layers = [64]
g_layers = [64]

q0_sigma_init, q0_sigma_min = 5, 1
q1_sigma_init, q1_sigma_min = 5, 1
q2_sigma_init, q2_sigma_min = 5, 1
f_sigma_init, f_sigma_min = 5, 1
g_sigma_init, g_sigma_min = 5, 1

# Normalizing Flow (NF)
q1_flow_layers  = 2
f_flow_layers   = 2
flow_sample_num = 100
flow_type       = "MAF"

# bidirectional RNN
y_smoother_Dhs = [64]
X0_smoother_Dhs = [64]

# ----------------------- SSM flags ------------------------ #

# if q1 and f share the same network
# (ATTENTION: even if use_2_q == True, f and q1 can still use different networks)
use_bootstrap = True

# should q use true_X to sample? (useful for debugging)
q_uses_true_X = False

# if q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t)
# if True, q_uses_true_X will be overwritten as False
use_2_q = True

# whether use input in q and f
use_input = False

# whether emission uses Poisson distribution
poisson_emission = False

# whether transitions (q1 and f) use Normalizing Flow
flow_transition = False

# ----------------------- FFN flags ------------------------ #

# if f and q use residual
use_residual = False

# if q, f and g networks also output covariance (sigma)
output_cov = False

# if q, f and g networks also output covariance (sigma)
diag_cov = False

# dropout rate for FFN
dropout_rate = 0.0

# ----------------------- TFS flags ------------------------ #
# whether use Two Filter Smoothing
TFS = False

# whether backward filtering in TFS uses different q0
TFS_use_diff_q0 = True

# ----------------------- FFBS flags ----------------------- #
# whether use Forward Filtering Backward Smoothing
FFBS = False

# how fast the model transfers from filtering to smoothing
smoothing_perc_factor = 2

# whether use smoothing for inference or leaning
FFBS_to_learn = False

# --------------------- smoother flags --------------------- #
# whether smooth observations with birdectional RNNs
smooth_obs = True

# whether use a separate RNN for getting X0
X0_use_separate_RNN = True

# whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn or tf.nn.bidirectional_dynamic_rnn
# check https://stackoverflow.com/a/50552539 for differences between them
use_stack_rnn = True

# --------------------- training flags --------------------- #

# stop training early if validation set does not improve
early_stop_patience = 200

# reduce learning rate when testing loss doesn't improve for some time
lr_reduce_patience = 30

# the factor to reduce lr, new_lr = old_lr * lr_reduce_factor
lr_reduce_factor = 1 / np.sqrt(2)

# minimum lr
min_lr = lr / 10

# The clipping ratio of gradient based on global L2 norm
clip_norm = 10.0

# --------------------- printing and data saving params --------------------- #
# frequency to evaluate testing loss & other metrics and save results
print_freq = 5

# whether to save the followings during training
#   hidden trajectories
#   k-step y-hat

save_trajectory = True
save_y_hat = True

# dir to save all results
rslt_dir_name = "Allen_wI"

# number of steps to predict y-hat and calculate R_square
MSE_steps = 30

# lattice shape [# of rows, # of columns] to draw arrows in quiver plot
lattice_shape = [25, 25]

# number of testing data used to save hidden trajectories, y-hat, gradient and etc
# will be clipped by number of testing data
saving_num = 30

# whether to save tensorboard
save_tensorboard = False

# whether to save model
save_model = False

q0_layers = ",".join([str(x) for x in q0_layers])
q1_layers = ",".join([str(x) for x in q1_layers])
q2_layers = ",".join([str(x) for x in q2_layers])
f_layers = ",".join([str(x) for x in f_layers])
g_layers = ",".join([str(x) for x in g_layers])
y_smoother_Dhs = ",".join([str(x) for x in y_smoother_Dhs])
X0_smoother_Dhs = ",".join([str(x) for x in X0_smoother_Dhs])
lattice_shape = ",".join([str(x) for x in lattice_shape])


# ================================================ tf.flags ================================================ #

flags = tf.app.flags


# --------------------- training hyperparameters --------------------- #
flags.DEFINE_integer("Dx", Dx, "dimension of hidden states")
flags.DEFINE_integer("Dy", Dy, "dimension of observations")
flags.DEFINE_integer("Di", Di, "dimension of inputs")

flags.DEFINE_integer("n_particles", n_particles, "number of particles")
flags.DEFINE_integer("batch_size", batch_size, "batch_size")
flags.DEFINE_float("lr", lr, "learning rate")
flags.DEFINE_integer("epoch", epoch, "number of epoch")

flags.DEFINE_integer("seed", seed, "random seed for np.random and tf")


# --------------------- data set parameters --------------------- #

flags.DEFINE_boolean("generateTrainingData", generateTrainingData, "True: generate data set from simulation; "
                                                                   "False: read data set from the file")
flags.DEFINE_string("datadir", datadir, "path of the data set file")
flags.DEFINE_string("datadict", datadict, "name of the data set file")
flags.DEFINE_boolean("isPython2", isPython2, "Was the data pickled in python 2?")


flags.DEFINE_integer("time", time, "number of timesteps for simulated data")
flags.DEFINE_integer("n_train", n_train, "number of trajactories for traning set")
flags.DEFINE_integer("n_test", n_test, "number of trajactories for testing set")


# --------------------- model parameters --------------------- #
# Feed-Forward Network (FFN) architectures
flags.DEFINE_string("q0_layers", q0_layers, "architecture for q0 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("q1_layers", q1_layers, "architecture for q1 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("q2_layers", q2_layers, "architecture for q2 network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("f_layers",  f_layers,  "architecture for f network, int seperated by comma, "
                                            "for example: '50,50' ")
flags.DEFINE_string("g_layers",  g_layers,  "architecture for g network, int seperated by comma, "
                                            "for example: '50,50' ")

flags.DEFINE_float("q0_sigma_init", q0_sigma_init, "initial value of q0_sigma")
flags.DEFINE_float("q1_sigma_init", q1_sigma_init, "initial value of q1_sigma")
flags.DEFINE_float("q2_sigma_init", q2_sigma_init, "initial value of q2_sigma")
flags.DEFINE_float("f_sigma_init",  f_sigma_init,  "initial value of f_sigma")
flags.DEFINE_float("g_sigma_init",  g_sigma_init,  "initial value of g_sigma")

flags.DEFINE_float("q0_sigma_min", q0_sigma_min, "minimal value of q0_sigma")
flags.DEFINE_float("q1_sigma_min", q1_sigma_min, "minimal value of q1_sigma")
flags.DEFINE_float("q2_sigma_min", q2_sigma_min, "minimal value of q2_sigma")
flags.DEFINE_float("f_sigma_min",  f_sigma_min,  "minimal value of f_sigma")
flags.DEFINE_float("g_sigma_min",  g_sigma_min,  "minimal value of g_sigma")

# Normalizing Flow
flags.DEFINE_integer("q1_flow_layers",  q1_flow_layers,  "number of layers of q1 normalizing flow")
flags.DEFINE_integer("f_flow_layers",   f_flow_layers,   "number of layers of f normalizing flow")
flags.DEFINE_integer("flow_sample_num", flow_sample_num, "number of samples used to determine the mean of flow")
flags.DEFINE_string("flow_type",        flow_type,       "type of flow to use: MAF, IAF or RealNVP")

# bidirectional RNN
flags.DEFINE_string("y_smoother_Dhs", y_smoother_Dhs, "number of units for y_smoother birdectional RNNs, "
                                                      "int seperated by comma")
flags.DEFINE_string("X0_smoother_Dhs", X0_smoother_Dhs, "number of units for X0_smoother birdectional RNNs, "
                                                        "int seperated by comma")

# --------------------- SSM flags --------------------- #
flags.DEFINE_boolean("use_bootstrap", use_bootstrap, "whether q1 and f share the same network, "
                                                     "(ATTENTION: even if use_2_q == True, "
                                                     "f and q1 can still use different networks)")
flags.DEFINE_boolean("q_uses_true_X", q_uses_true_X, "whether q1 uses true hidden states to sample")
flags.DEFINE_boolean("use_2_q", use_2_q, "whether q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t), "
                                         "if True, q_uses_true_X will be overwritten as False")
flags.DEFINE_boolean("use_input", use_input, "whether use input in q and f")
flags.DEFINE_boolean("flow_transition", flow_transition, "whether transitions (q1 and f) use Normalizing Flow")
flags.DEFINE_boolean("poisson_emission", poisson_emission, "whether emission uses Poisson distribution")
# --------------------- FFN flags --------------------- #
flags.DEFINE_boolean("use_residual", use_residual, "whether f and q use residual network")
flags.DEFINE_boolean("output_cov", output_cov, "whether q, f and g networks also output covariance (sigma)")
flags.DEFINE_boolean("diag_cov", diag_cov, "whether the networks only output diagonal value of cov matrix")
flags.DEFINE_float("dropout_rate", dropout_rate, "dropout rate for FFN")

# ----------------------- TFS flags ------------------------ #

flags.DEFINE_boolean("TFS", TFS, "whether use Two Filter Smoothing")
flags.DEFINE_boolean("TFS_use_diff_q0", TFS_use_diff_q0, "whether backward filtering in TFS uses different q0")

# ----------------------- FFBS flags ----------------------- #

flags.DEFINE_boolean("FFBS", FFBS, "whether use Forward Filtering Backward Smoothing")
flags.DEFINE_float("smoothing_perc_factor", smoothing_perc_factor,
                   "determine how the percentage of smoothing loss in the total loss changes with epoch num, "
                   "the percentage of smoothing loss = 1 - (1 - current_epoch / total_epoch) ** smoothing_perc_factor")
flags.DEFINE_boolean("FFBS_to_learn", FFBS_to_learn, "whether use FFBS for leaning or inference")

# --------------------- smoother flags --------------------- #

flags.DEFINE_boolean("smooth_obs", smooth_obs, "whether smooth observations with birdectional RNNs")
flags.DEFINE_boolean("X0_use_separate_RNN", X0_use_separate_RNN, "whether use a separate RNN for getting X0")
flags.DEFINE_boolean("use_stack_rnn", use_stack_rnn, "whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn "
                                                     "or tf.nn.bidirectional_dynamic_rnn")

# --------------------- training flags --------------------- #

flags.DEFINE_integer("early_stop_patience", early_stop_patience,
                     "stop training early if validation set does not improve for certain epochs")

flags.DEFINE_integer("lr_reduce_patience", lr_reduce_patience,
                     "educe learning rate when testing loss doesn't improve for some time")
flags.DEFINE_float("lr_reduce_factor", lr_reduce_factor,
                   "the factor to reduce learning rate, new_lr = old_lr * lr_reduce_factor")
flags.DEFINE_float("min_lr", min_lr, "minimum learning rate")
flags.DEFINE_float("clip_norm", clip_norm, "The clipping ratio of gradient based on global L2 norm")

# --------------------- printing and data saving params --------------------- #

flags.DEFINE_integer("print_freq", print_freq, "frequency to evaluate testing loss & other metrics and save results")

flags.DEFINE_boolean("save_trajectory", save_trajectory, "whether to save hidden trajectories during training")
flags.DEFINE_boolean("save_y_hat", save_y_hat, "whether to save k-step y-hat during training")

flags.DEFINE_string("rslt_dir_name", rslt_dir_name, "dir to save all results")
flags.DEFINE_integer("MSE_steps", MSE_steps, "number of steps to predict y-hat and calculate R_square")

flags.DEFINE_string("lattice_shape", lattice_shape, "lattice shape [# of rows, # of columns] "
                                                    "to draw arrows in quiver plot")

flags.DEFINE_integer("saving_num", saving_num, "number of testing data used to "
                                               "save hidden trajectories, y-hat, gradient and etc, "
                                               "will be clipped by number of testing data")

flags.DEFINE_boolean("save_tensorboard", save_tensorboard, "whether to save tensorboard")
flags.DEFINE_boolean("save_model", save_model, "whether to save model")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    tf.app.run()

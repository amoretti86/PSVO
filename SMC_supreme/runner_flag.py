import tensorflow as tf
import tensorflow_probability as tfp
import os

from runner import main

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # to avoid lots of log about the device

print("the code is written in:")
print("\ttensorflow version: 1.12.0")
print("\ttensorflow_probability version: 0.5.0")

print("the system uses:")
print("\ttensorflow version:", tf.__version__)
print("\ttensorflow_probability version:", tfp.__version__)


# --------------------- training hyperparameters --------------------- #
Dx = 3
Dy = 1
Di = 1
n_particles = 64

batch_size = 1
lr = 1e-3
epoch = 300
seed = 0

# --------------------- data set parameters --------------------- #
# generate synthetic data?
generateTrainingData = False

# if reading data from file
datadir = "C:/Users/admin/Desktop/research/code/VISMC/data/fhn/[1,0]_obs_cov_0.1/"
# "C:/Users/admin/Desktop/research/code/VISMC/data/lorenz/[1,0,0]_obs_cov_0.4/"
# "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/lorenz/[1,0,0]_obs_cov_0.4/"
datadict = "datadict"
isPython2 = False

# time, n_train and n_test will be overwritten if loading data from the file
time = 200
n_train = 200 * batch_size
n_test = 40 * batch_size

# --------------------- model parameters --------------------- #
# network architectures
q_train_layers = [50]
f_train_layers = [50]
g_train_layers = [50]
q2_train_layers = [50]

q_sigma_init, q_sigma_min = 5, 1
f_sigma_init, f_sigma_min = 5, 1
g_sigma_init, g_sigma_min = 5, 1
q2_sigma_init, q2_sigma_min = 5, 1

lstm_Dh = 10
X0_layers = [10]

# do q and f use the same network?
use_bootstrap = True

# should q use true_X to sample? (useful for debugging)
q_uses_true_X = False

# stop training early if validation set does not improve
maxNumberNoImprovement = 5

# if x0 is learnable or takes ground truth
x_0_learnable = False

# filtering or smoothing
smoothing = False

# if f and q use residual
use_residual = False

# if q, f and g networks also output covariance (sigma)
output_cov = False

# if q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t)
# if True, use_bootstrap will be overwritten as True
#          q_takes_y as False
#          q_uses_true_X as False
use_2_q = True

# whether use tf.stop_gradient when resampling and reweighting weights (during smoothing)
use_stop_gradient = False

# how fast the model transfers from filtering to smoothing
smoothing_perc_factor = 0

# whether use birdectional RNN to get X0 and encode observation
get_X0_w_bRNN = True
smooth_y_w_bRNN = True

# whether use input in q and f
use_input = False

# --------------------- printing and data saving params --------------------- #
print_freq = 5

store_res = True
rslt_dir_name = "lorenz_1D"
MSE_steps = 10

# how many trajectories to draw in quiver plot
quiver_traj_num = min(5, n_train, n_test)
lattice_shape = [25, 25]  # [25, 25] or [10, 10, 3]

saving_num = 40

save_tensorboard = False
save_model = False
save_freq = 10

q_train_layers = ",".join([str(x) for x in q_train_layers])
f_train_layers = ",".join([str(x) for x in f_train_layers])
g_train_layers = ",".join([str(x) for x in g_train_layers])
q2_train_layers = ",".join([str(x) for x in q2_train_layers])
X0_layers = ",".join([str(x) for x in X0_layers])
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

flags.DEFINE_string("q_train_layers", q_train_layers, "architecture for q network, int seperated by comma, "
                                                      "for example: '50,50' ")
flags.DEFINE_string("f_train_layers", f_train_layers, "architecture for f network, int seperated by comma, "
                                                      "for example: '50,50' ")
flags.DEFINE_string("g_train_layers", g_train_layers, "architecture for g network, int seperated by comma, "
                                                      "for example: '50,50' ")
flags.DEFINE_string("q2_train_layers", q2_train_layers, "architecture for q2 network, int seperated by comma, "
                                                        "for example: '50,50' ")
flags.DEFINE_integer("lstm_Dh", lstm_Dh, "hidden state dimension for bidirectional LSTM")
flags.DEFINE_string("X0_layers", X0_layers, "architecture for X0 network, int seperated by comma, "
                                            "for example: '50,50' ")

flags.DEFINE_float("q_sigma_init", q_sigma_init, "initial value of q_sigma")
flags.DEFINE_float("f_sigma_init", f_sigma_init, "initial value of f_sigma")
flags.DEFINE_float("g_sigma_init", g_sigma_init, "initial value of g_sigma")
flags.DEFINE_float("q2_sigma_init", q2_sigma_init, "initial value of q2_sigma")

flags.DEFINE_float("q_sigma_min", q_sigma_min, "minimal value of q_sigma")
flags.DEFINE_float("f_sigma_min", f_sigma_min, "minimal value of f_sigma")
flags.DEFINE_float("g_sigma_min", g_sigma_min, "minimal value of g_sigma")
flags.DEFINE_float("q2_sigma_min", q2_sigma_min, "minimal value of q2_sigma")

flags.DEFINE_boolean("use_bootstrap", use_bootstrap, "whether use f and q are the same")
flags.DEFINE_boolean("q_uses_true_X", q_uses_true_X, "whether q uses true hidden states to sample")

flags.DEFINE_integer("maxNumberNoImprovement", maxNumberNoImprovement,
                     "stop training early if validation set does not improve for certain epochs")

flags.DEFINE_boolean("x_0_learnable", x_0_learnable, "whether x0 is learnable or takes ground truth")
flags.DEFINE_boolean("smoothing", smoothing, "whether filtering or smoothing")
flags.DEFINE_boolean("use_residual", use_residual, "whether f and q use residual network")
flags.DEFINE_boolean("output_cov", output_cov, "whether q, f and g networks also output covariance (sigma)")
flags.DEFINE_boolean("use_2_q", use_2_q, "whether q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t), "
                                         "if True, use_bootstrap will be overwritten as True, "
                                         "q_takes_y as False, "
                                         "q_uses_true_X as False")
flags.DEFINE_boolean("use_stop_gradient", use_stop_gradient, "whether use tf.stop_gradient "
                                                             "when resampling and reweighting weights during smoothing")

flags.DEFINE_float("smoothing_perc_factor", smoothing_perc_factor,
                   "determine how the percentage of smoothing loss in the total loss changes with epoch num, "
                   "the percentage of smoothing loss = 1 - (1 - current_epoch / total_epoch) ** smoothing_perc_factor")

flags.DEFINE_boolean("get_X0_w_bRNN", get_X0_w_bRNN, "whether learn X0 from obs with a bidirectional RNN")
flags.DEFINE_boolean("smooth_y_w_bRNN", smooth_y_w_bRNN, "whether encode obs with a bidirectional RNN")

flags.DEFINE_boolean("use_input", use_input, "whether use input in q and f")

# --------------------- printing and data saving params --------------------- #

flags.DEFINE_integer("print_freq", print_freq, "frequency to print log during training")
flags.DEFINE_boolean("store_res", store_res, "whether store results")

flags.DEFINE_string("rslt_dir_name", rslt_dir_name, "name of the dir storing the results")
flags.DEFINE_integer("MSE_steps", MSE_steps, "number of steps to predict MSE and R_square")

flags.DEFINE_integer("quiver_traj_num", quiver_traj_num, "frequency to print log during training")
flags.DEFINE_string("lattice_shape", lattice_shape, "frequency to print log during training")

flags.DEFINE_integer("saving_num", saving_num, "frequency to print log during training")

flags.DEFINE_boolean("save_tensorboard", save_tensorboard, "frequency to print log during training")
flags.DEFINE_boolean("save_model", save_model, "frequency to print log during training")
flags.DEFINE_integer("save_freq", save_freq, "frequency to print log during training")

FLAGS = flags.FLAGS

if __name__ == "__main__":
    tf.app.run()

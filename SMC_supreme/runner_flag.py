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
n_particles = 500

batch_size = 1
lr = 1e-3
epoch = 300
seed = 0

# --------------------- data set parameters --------------------- #
# generate synthetic data?
generateTrainingData = False

# if reading data from file
datadir = "/ifs/scratch/c2b2/ip_lab/zw2504/VISMC/data/lorenz/[1,0,0]_obs_cov_0.4"
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

q_sigma_init, q_sigma_min = 5, 1
f_sigma_init, f_sigma_min = 5, 1
g_sigma_init, g_sigma_min = 5, 1

# do q and f use the same network?
use_bootstrap = True

# if q takes y_t as input
# if use_bootstrap, q_takes_y will be overwritten as False
q_takes_y = False

# should q use true_X to sample? (useful for debugging)
q_uses_true_X = False

# term to weight the added contribution of the MSE to the cost
loss_beta = 0.0

# stop training early if validation set does not improve
maxNumberNoImprovement = 5

# if x0 is learnable or takes ground truth
x_0_learnable = False

# if f and q use residual
use_residual = False

# if q, f and g networks also output covariance (sigma)
output_cov = False

# --------------------- printing and data saving params --------------------- #
print_freq = 5

store_res = True
rslt_dir_name = "lorenz_1D"
MSE_steps = min(10, time - 1)

# how many trajectories to draw in quiver plot
quiver_traj_num = min(5, n_train, n_test)
lattice_shape = [10, 10, 3]  # [25, 25] or [10, 10, 3]

saving_num = min(10, n_train, n_test)

save_tensorboard = False
save_model = False
save_freq = 10

q_train_layers = ",".join([str(x) for x in q_train_layers])
f_train_layers = ",".join([str(x) for x in f_train_layers])
g_train_layers = ",".join([str(x) for x in g_train_layers])
lattice_shape = ",".join([str(x) for x in lattice_shape])


# ================================================ tf.flags ================================================ #

flags = tf.app.flags


# --------------------- training hyperparameters --------------------- #
flags.DEFINE_integer("Dx", Dx, "dimension of hidden states")
flags.DEFINE_integer("Dy", Dy, "dimension of observations")

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

flags.DEFINE_float("q_sigma_init", q_sigma_init, "initial value of q_sigma")
flags.DEFINE_float("f_sigma_init", f_sigma_init, "initial value of f_sigma")
flags.DEFINE_float("g_sigma_init", g_sigma_init, "initial value of g_sigma")

flags.DEFINE_float("q_sigma_min", q_sigma_min, "minimal value of q_sigma")
flags.DEFINE_float("f_sigma_min", f_sigma_min, "minimal value of f_sigma")
flags.DEFINE_float("g_sigma_min", g_sigma_min, "minimal value of g_sigma")

flags.DEFINE_boolean("use_bootstrap", use_bootstrap, "whether use f and q are the same")
flags.DEFINE_boolean("q_takes_y", q_takes_y, "whether input of q stack y")
flags.DEFINE_boolean("q_uses_true_X", q_uses_true_X, "whether q uses true hidden states to sample")

flags.DEFINE_float("loss_beta", loss_beta, "loss = log_ZSMC + loss_beta * MSE_0_step")
flags.DEFINE_integer("maxNumberNoImprovement", maxNumberNoImprovement,
                     "stop training early if validation set does not improve for certain epochs")

flags.DEFINE_boolean("x_0_learnable", x_0_learnable, "if x0 is learnable or takes ground truth")
flags.DEFINE_boolean("use_residual", use_residual, "if f and q use residual network")
flags.DEFINE_boolean("output_cov", output_cov, "if q, f and g networks also output covariance (sigma)")


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

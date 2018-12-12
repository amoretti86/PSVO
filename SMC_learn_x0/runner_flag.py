import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

# for data saving stuff
import pickle
import json
import os
import pdb

# import from files
from transformation.fhn import fhn_transformation, tf_fhn_transformation
from transformation.linear import linear_transformation, tf_linear_transformation
from transformation.lorenz import lorenz_transformation, tf_lorenz_transformation
from transformation.MLP import MLP_transformation

from distribution.dirac_delta import dirac_delta
from distribution.mvn import mvn, tf_mvn
from distribution.poisson import poisson, tf_poisson

from rslts_saving.rslts_saving import *
from rslts_saving.fhn_rslts_saving import *
from rslts_saving.lorenz_rslts_saving import *
from trainer import trainer

from encoder import encoder_cell
from sampler import create_dataset
from SMC import SMC

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # to avoid lots of log about the device

print("the code is written in:")
print("\ttensorflow version: 1.12.0")
print("\ttensorflow_probability version: 0.5.0")

print("the system uses:")
print("\ttensorflow version:", tf.__version__)
print("\ttensorflow_probability version:", tfp.__version__)

flags = tf.app.flags
flags.DEFINE_integer("Dx", 2, "dimension of hidden states")
flags.DEFINE_integer("Dy", 1, "dimension of hidden observations")

flags.DEFINE_integer("n_particles", 500, "number of particles")
flags.DEFINE_integer("batch_size", 5, "batch_size")
flags.DEFINE_float("lr", 1e-3, "learning rate")
flags.DEFINE_integer("epoch", 300, "number of epoch")

flags.DEFINE_integer("seed", 0, "random seed for np.random and tf")

flags.DEFINE_integer("time", 200, "number of timesteps for simulated data")
flags.DEFINE_integer("n_train", 200, "number of trajactories for traning set")
flags.DEFINE_integer("n_test", 40, "number of trajactories for testing set")

flags.DEFINE_string("q_train_layers", "50", "architecture for q network, int seperated by comma, example: 50,50")
flags.DEFINE_string("f_train_layers", "50", "architecture for f network, int seperated by comma, example: 50,50")
flags.DEFINE_string("g_train_layers", "50", "architecture for g network, int seperated by comma, example: 50,50")
flags.DEFINE_float("q_sigma_init", 5, "initial value of q_sigma")
flags.DEFINE_float("f_sigma_init", 5, "initial value of f_sigma")
flags.DEFINE_float("g_sigma_init", 5, "initial value of g_sigma")
flags.DEFINE_float("q_sigma_min", 1, "minimal value of q_sigma")
flags.DEFINE_float("f_sigma_min", 1, "minimal value of f_sigma")
flags.DEFINE_float("g_sigma_min", 1, "minimal value of g_sigma")

flags.DEFINE_boolean("use_bootstrap", True, "whether use f and q are the same")
flags.DEFINE_boolean("q_takes_y", True, "whether input of q stack y")
flags.DEFINE_boolean("q_uses_true_X", False, "whether q uses true hidden states to sample")

flags.DEFINE_float("loss_beta", 0.0, "loss = log_ZSMC + loss_beta * MSE_0_step")
flags.DEFINE_integer("maxNumberNoImprovement", 5, "When to stop training")

flags.DEFINE_boolean("generateTrainingData", False, "True: generate data set from simulation; "
                                                    "False: read data set from the file")
flags.DEFINE_string("datadir", "Not defined", "path of the data set file")
flags.DEFINE_string("datadict", "datadict", "name of the data set file")
flags.DEFINE_boolean("isPython2", False, "Was the data pickled in python 2?")

flags.DEFINE_integer("print_freq", 5, "frequency to print log during training")
flags.DEFINE_integer("MSE_steps", 10, "number of steps to predict MSE and R_square")

flags.DEFINE_integer("store_res", True, "whether store results")
flags.DEFINE_string("rslt_dir_name", "rslts", "name of the dir storing the results")


def main():
    # ============================================ parameter part ============================================ #
    # training hyperparameters
    Dx = flags.Dx
    Dy = flags.Dy
    n_particles = flags.n_particles

    batch_size = flags.batch_size
    lr = flags.lr
    epoch = flags.epoch
    seed = flags.seed
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # time, n_train and n_test will be overwritten if loading data from the file
    time = flags.time
    n_train = flags.n_train
    n_test = flags.n_test
    assert n_train % batch_size == 0
    assert n_test % batch_size == 0

    # Define encoder and decoder network architectures
    q_train_layers = [int(x) for x in flags.q_train_layers.split(",")]
    f_train_layers = [int(x) for x in flags.f_train_layers.split(",")]
    g_train_layers = [int(x) for x in flags.g_train_layers.split(",")]

    q_sigma_init, q_sigma_min = flags.q_sigma_init, flags.q_sigma_min
    f_sigma_init, f_sigma_min = flags.f_sigma_init, flags.f_sigma_min
    g_sigma_init, g_sigma_min = flags.g_sigma_init, flags.g_sigma_min

    # do q and f use the same network?
    use_bootstrap = flags.use_bootstrap

    # if q takes y_t as input
    # if is_bootstrap, q_takes_y will be overwritten as False
    q_takes_y = flags.q_takes_y

    # should q use true_X to sample? (useful for debugging)
    q_uses_true_X = flags.q_uses_true_X

    # term to weight the added contribution of the MSE to the cost
    loss_beta = flags.loss_beta

    # stop training early if validation set does not improve
    maxNumberNoImprovement = flags.maxNumberNoImprovement

    # generate synthetic data?
    generateTrainingData = flags.generateTrainingData

    # if reading data from file
    datadir = flags.datadir
    datadict = flags.datadict
    isPython2 = flags.isPython2

    # printing and data saving params
    print_freq = flags.print_freq

    store_res = flags.store_res
    MSE_steps = min(flags.MSE_steps, time - 1)
    save_freq = 10
    saving_num = min(n_train, 2 * batch_size)
    rslt_dir_name = flags.rslt_dir_name

    # ============================================= dataset part ============================================= #
    if generateTrainingData:

        # integrate differential equations to simulate the FHN or Lorenz systems
        sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
        f_params = (sigma, rho, beta, dt)

        # a, b, c, I, dt = 1.0, 0.95, 0.05, 1.0, 0.15
        # f_params = (a, b, c, I, dt)

        f_sample_cov = 0.0 * np.eye(Dx)

        # g_params = np.random.randn(Dy, Dx)  # np.array([[1.0, 1.0]]) or np.random.randn(Dy, Dx)
        g_params = np.array([[1.0, 0.0, 1.0]])
        g_sample_cov = 0.1 * np.eye(Dy)

        # f_sample_tran = fhn_transformation(f_params)
        f_sample_tran = lorenz_transformation(f_params)
        f_sample_dist = dirac_delta(f_sample_tran)

        g_sample_tran = linear_transformation(g_params)
        g_sample_dist = mvn(g_sample_tran, g_sample_cov)

        true_model_dict = {"f_params": f_params,
                           "f_cov": f_sample_cov,
                           "g_params": g_params,
                           "g_cov": g_sample_cov}

        hidden_train, obs_train, hidden_test, obs_test = \
            create_dataset(n_train, n_test, time, Dx, Dy, f_sample_dist, g_sample_dist, lb=-2.5, ub=2.5)
        print("finished creating dataset")

    else:
        # load data
        with open(datadir + datadict, 'rb') as handle:
            if isPython2:
                data = pickle.load(handle, encoding='latin1')
            else:
                data = pickle.load(handle)

        obs_train = data['Ytrain']
        obs_test = data['Yvalid']

        n_train = obs_train.shape[0]
        n_test = obs_test.shape[0]
        time = obs_train.shape[1]

        hidden_train = data['Xtrue'][:n_train]
        hidden_test = data['Xtrue'][n_train:]

        print("finished loading dataset")

    # ============================================== model part ============================================== #
    # placeholders
    all_x_0 = tf.Variable(np.zeros((n_train + n_test, Dx)), dtype=tf.float32, name="all_x_0")
    x_0_idx = tf.placeholder(tf.int32, shape=(batch_size), name="x_0_idx")
    x_0 = tf.gather(all_x_0, x_0_idx, name="x_0")
    obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name="obs")
    hidden = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name="hidden")

    q_train_tran = MLP_transformation(q_train_layers, Dx, name='q_train_tran')
    g_train_tran = MLP_transformation(g_train_layers, Dy, name="g_train_tran")
    if use_bootstrap:
        f_train_tran = q_train_tran
        q_takes_y = False
    else:
        f_train_tran = MLP_transformation(f_train_layers, Dx, name="f_train_tran")

    q_train_dist = tf_mvn(q_train_tran, x_0, sigma_init=q_sigma_init, sigma_min=q_sigma_min, name="q_train_dist")
    g_train_dist = tf_mvn(g_train_tran, None, sigma_init=g_sigma_init, sigma_min=g_sigma_min, name="g_train_dist")
    if use_bootstrap:
        f_train_dist = q_train_dist
    else:
        f_train_dist = tf_mvn(f_train_tran, x_0, sigma_init=f_sigma_init, sigma_min=f_sigma_min, name="f_train_dist")

    init_dict = {"q_sigma_init": q_sigma_init,
                 "q_sigma_min": q_sigma_min,
                 "f_sigma_init": f_sigma_init,
                 "f_sigma_min": f_sigma_min,
                 "g_sigma_init": g_sigma_init,
                 "g_sigma_min": g_sigma_min}

    SMC_train = SMC(q_train_dist, f_train_dist, g_train_dist,
                    n_particles, batch_size,
                    q_takes_y=q_takes_y,
                    q_uses_true_X=q_uses_true_X,
                    name="log_ZSMC_train")

    # =========================================== data saving part =========================================== #
    if store_res:
        Experiment_params = {"n_particles": n_particles,
                             "time": time,
                             "batch_size": batch_size,
                             "lr": lr,
                             "epoch": epoch,
                             "seed": seed,
                             "n_train": n_train,
                             "bootstrap": use_bootstrap,
                             "q_take_y": q_takes_y,
                             "use_true_X": q_uses_true_X,
                             "beta": loss_beta,
                             "rslt_dir_name": rslt_dir_name}
        print("Experiment_params")
        for key, val in Experiment_params.items():
            print("\t{}: {}".format(key, val))

        RLT_DIR = create_RLT_DIR(Experiment_params)
        print("RLT_DIR:", RLT_DIR)

    # ============================================= training part ============================================ #
    mytrainer = trainer(Dx, Dy,
                        n_particles, time,
                        batch_size, lr, epoch,
                        MSE_steps,
                        store_res,
                        loss_beta, maxNumberNoImprovement)

    mytrainer.set_SMC(SMC_train)
    mytrainer.set_placeholders(x_0_idx, obs, hidden)

    if store_res:
        mytrainer.set_rslt_saving(RLT_DIR, save_freq, saving_num)
        if Dx == 2:
            lattice = tf.placeholder(tf.float32, shape=(50, 50, 2), name="lattice")
            nextX = SMC_train.get_nextX(lattice)
            mytrainer.set_quiver_arg(nextX, lattice)

        if Dx == 3:
            lattice = tf.placeholder(tf.float32, shape=(10, 10, 3, 3), name="lattice")
            nextX = SMC_train.get_nextX(lattice)
            mytrainer.set_quiver_arg(nextX, lattice)

    mytrainer.set_fitReal(not generateTrainingData)
    losses, tensors = mytrainer.train(obs_train, obs_test, print_freq, hidden_train, hidden_test)

    # ======================================= another data saving part ======================================= #
    if store_res:
        log_ZSMC_trains, log_ZSMC_tests, \
            MSE_trains, MSE_tests, \
            R_square_trains, R_square_tests = losses
        log_train, ys_hat = tensors

        Xs = log_train[0]
        Xs_val = mytrainer.evaluate(Xs, {obs: obs_train[0:saving_num],
                                         x_0_idx: np.arange(saving_num),
                                         hidden: hidden_train[0:saving_num]})
        ys_hat_val = mytrainer.evaluate(ys_hat, {obs: obs_train[0:saving_num],
                                                 x_0_idx: np.arange(saving_num),
                                                 hidden: hidden_train[0:saving_num]})

        print("finish evaluating training results")

        plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num=saving_num)
        plot_learning_results(RLT_DIR, Xs_val, hidden_train, saving_num=saving_num)
        plot_y_hat(RLT_DIR, ys_hat_val, obs_train, saving_num=saving_num)

        if Dx == 2:
            plot_fhn_results(RLT_DIR, Xs_val)

        if Dx == 3:
            plot_lorenz_results(RLT_DIR, Xs_val)

        params_dict = {"time": time,
                       "n_particles": n_particles,
                       "batch_size": batch_size,
                       "lr": lr,
                       "epoch": epoch,
                       "n_train": n_train,
                       "seed": seed,
                       "use_bootstrap": use_bootstrap,
                       "q_takes_y": q_takes_y,
                       "q_uses_true_X": q_uses_true_X,
                       "beta": loss_beta}
        loss_dict = {"log_ZSMC_trains": log_ZSMC_trains,
                     "log_ZSMC_tests": log_ZSMC_tests,
                     "MSE_trains": MSE_trains,
                     "MSE_tests": MSE_tests,
                     "R_square_trains": R_square_trains,
                     "R_square_tests": R_square_tests}
        data_dict = {"params": params_dict,
                     "init_dict": init_dict,
                     "loss": loss_dict}

        if generateTrainingData:
            data_dict["true_model_dict"] = true_model_dict

        with open(RLT_DIR + "data.json", "w") as f:
            json.dump(data_dict, f, indent=4, cls=NumpyEncoder)
        train_data_dict = {"hidden_train": hidden_train[0:saving_num],
                           "obs_train": obs_train[0:saving_num]}
        learned_model_dict = {"Xs_val": Xs_val,
                              "ys_hat_val": ys_hat_val}
        data_dict["train_data_dict"] = train_data_dict
        data_dict["learned_model_dict"] = learned_model_dict

        with open(RLT_DIR + "data.p", "wb") as f:
            pickle.dump(data_dict, f)

        plot_MSEs(RLT_DIR, MSE_trains, MSE_tests, print_freq)
        plot_R_square(RLT_DIR, R_square_trains, R_square_tests, print_freq)
        plot_log_ZSMC(RLT_DIR, log_ZSMC_trains, log_ZSMC_tests, print_freq)


if __name__ == "__main__":
    tf.app.run()

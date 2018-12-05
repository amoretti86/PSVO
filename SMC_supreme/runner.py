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

if __name__ == "__main__":

    # ============================================ parameter part ============================================ #
    # training hyperparameters
    Dx = 2
    Dy = 1
    n_particles = 50
    time = 200

    batch_size = 5
    lr = 1e-3
    epoch = 10
    seed = 0
    tf.set_random_seed(seed)
    np.random.seed(seed)

    n_train = 18 * batch_size
    n_test = 1  * batch_size

    # Define encoder and decoder network architectures
    q_train_layers = [20]
    f_train_layers = [20]
    g_train_layers = [20]

    # do q and f use the same network?
    use_bootstrap = True
    # if q takes y_t as input
    # if is_bootstrap, q_takes_y will be overwritten as False
    q_takes_y = False
    # should q use true_X to sample? (useful for debugging)
    q_use_true_X = False
    # scale observation by the mean abs value of obs?
    scale_obs = False

    # term to weight the added contribution of the MSE to the cost
    beta = 0.5
    # stop training early if validation set does not improve
    maxNumberNoImprovement = 5

    # generate synthetic data?
    generateTrainingData = False
    # pass ground truth initial state
    passInitialState = False

    # if reading data from file
    datadir = '/Users/antoniomoretti/Desktop/dhern-ts_wcommona-b4b1ad88b3aa/data/fitzhughnagumo/'
    datadict = 'datadict'
    isPython2 = True

    # priors on initial state x_0
    # TODO: This should be revisited, must act on trainable variables in runner and trainer
    x_0_init_mean = 0.0
    x_0_init_scale = 1.0

    # printing and data saving params
    print_freq = 10

    store_res = True
    MSE_steps = 5
    save_freq = 10
    saving_num = min(n_train, 2 * batch_size)
    rslt_dir_name = "FN_1D_obs"

    # ============================================== model part ============================================== #
    if generateTrainingData:

        # integrate differential equations to simulate the FHN or Lorenz systems
        # sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
        # f_params = (sigma, rho, beta, dt)

        a, b, c, I, dt = 1.0, 0.95, 0.05, 1.0, 0.15
        f_params = (a, b, c, I, dt)

        f_sample_cov = 0.0 * np.eye(Dx)

        #g_params = np.random.randn(Dy, Dx)  # np.array([[1.0, 1.0]]) or np.random.randn(Dy, Dx)
        g_params = np.array([[1.0, 0.0]])
        g_sample_cov = 0.1 * np.eye(Dy)

        # transformation can be: fhn_transformation, linear_transformation, lorenz_transformation
        # distribution can be: dirac_delta, mvn, poisson
        f_sample_tran = fhn_transformation(f_params)
        # f_sample_tran = lorenz_transformation(f_params)
        f_sample_dist = dirac_delta(f_sample_tran)

        g_sample_tran = linear_transformation(g_params)
        g_sample_dist = mvn(g_sample_tran, g_sample_cov)

        true_model_dict = {"f_params": f_params,
                           "f_cov": f_sample_cov,
                           "g_params": g_params,
                           "g_cov": g_sample_cov}

    # for training
    x_0 = tf.placeholder(tf.float32, shape=(batch_size, Dx), name="x_0")

    my_encoder_cell = None
    #f_train_tran = MLP_transformation(f_train_layers, Dx, name="f_train_tran")
    q_train_tran = MLP_transformation(q_train_layers, Dx, name='q_train_tran')
    g_train_tran = MLP_transformation(g_train_layers, Dy, name="g_train_tran")
    flow_tran = q_train_tran
    if use_bootstrap:
        f_train_tran = q_train_tran
        q_takes_y = False
    else:
        q_train_tran = MLP_transformation(q_train_layers, Dx, name="q_train_tran")

    # my_encoder_cell = encoder_cell(Dx, Dy, batch_size, time, name = "encoder_cell")
    # q_train_tran = my_encoder_cell.q_transformation
    # f_train_tran = my_encoder_cell.f_transformation

    q_sigma_init, q_sigma_min = 5, 1
    f_sigma_init, f_sigma_min = 5, 1
    g_sigma_init, g_sigma_min = 5, 1

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

    if generateTrainingData:

        # for evaluating log_ZSMC_true
        q_A = tf.constant(np.eye(Dx), dtype=tf.float32)
        q_cov = 1 * tf.eye(Dx)

        f_cov = 1 * tf.eye(Dx)

        g_A = tf.constant(g_params, dtype=tf.float32)
        g_cov = tf.constant(g_sample_cov, dtype=tf.float32)

        q_true_tran = tf_linear_transformation(q_A)
        q_true_dist = tf_mvn(q_true_tran, x_0, sigma=q_cov, name="q_true_dist")
        f_true_tran = tf_fhn_transformation(f_params)
        # f_true_tran = tf_lorenz_transformation(f_params)
        f_true_dist = tf_mvn(f_true_tran, x_0, sigma=f_cov, name="f_true_dist")
        g_true_tran = tf_linear_transformation(g_A)
        g_true_dist = tf_mvn(g_true_tran, name="g_true_dist")

    # =========================================== data saving part =========================================== #
    if store_res:
        Experiment_params = {"n_particles": n_particles,
                             "time": time,
                             "batch_size": batch_size,
                             "lr": lr,
                             "epoch": epoch,
                             "seed": seed,
                             "n_train": n_train,
                             "use_bootstrap": use_bootstrap,
                             "q_takes_y": q_takes_y,
                             "q_use_true_X": q_use_true_X,
                             "rslt_dir_name": rslt_dir_name}
        print("Experiment_params")
        for key, val in Experiment_params.items():
            print("\t{}: {}".format(key, val))

        RLT_DIR = create_RLT_DIR(Experiment_params)
        print("RLT_DIR:", RLT_DIR)

    # ============================================= dataset part ============================================= #
    if generateTrainingData == True:

        # Create train and test dataset
        hidden_train, obs_train, hidden_test, obs_test = \
            create_dataset(n_train, n_test, time, Dx, Dy, f_sample_dist, g_sample_dist, lb=-2.5, ub=2.5)
        print("finished creating dataset")


    if generateTrainingData == False:
        # load data
        with open(datadir + "datadict", 'rb') as handle:
            if isPython2:
                data = pickle.load(handle, encoding='latin1')
            else:
                data = pickle.load(handle)
        hidden_train = data['Xtrue'][0:80]
        hidden_valid = data['Xtrue'][80:100]
        obs_train = data['Ytrain']
        obs_test  = data['Yvalid']
        n_train = obs_train.shape[0]
        n_test = obs_test.shape[0]
        print("finished loading dataset")
        log_ZSMC_true_val = None

    if scale_obs:
        obs_all = abs(np.concatenate([obs_train, obs_test]))
        obs_train /= np.mean(obs_all, axis=(0, 1))
        obs_test /= np.mean(obs_all, axis=(0, 1))


    # ========================================== another model part ========================================== #
    # placeholders
    obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name="obs")
    hidden = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name="hidden")

    if generateTrainingData:
        SMC_true = SMC(q_true_dist, f_true_dist, g_true_dist,
                       n_particles, batch_size,
                       q_takes_y=False,
                       name="log_ZSMC_true")
    SMC_train = SMC(q_train_dist, f_train_dist, g_train_dist,
                    n_particles, batch_size,
                    encoder_cell=my_encoder_cell,
                    q_takes_y=q_takes_y,
                    q_use_true_X=q_use_true_X,
                    name="log_ZSMC_train")

    # ============================================= training part ============================================ #
    if True:
        mytrainer = trainer(Dx, Dy,
                            n_particles, time,
                            batch_size, lr, epoch,
                            MSE_steps,
                            store_res, beta, maxNumberNoImprovement,
                            x_0_init_mean, x_0_init_scale)

    mytrainer.set_placeholders(x_0, obs, hidden)
    mytrainer.set_rslt_saving(RLT_DIR, save_freq, saving_num)

    if generateTrainingData is False:
        mytrainer.set_SMC(None, SMC_train)
        mytrainer.set_fitReal(True)
    else:
        mytrainer.set_SMC(SMC_true, SMC_train)
        mytrainer.set_fitReal(False)

        if store_res:

            if isinstance(f_sample_tran, fhn_transformation) and my_encoder_cell is None:
                lattice = tf.placeholder(tf.float32, shape=(50, 50, 2), name="lattice")
                nextX = SMC_train.get_nextX(lattice)
                mytrainer.set_quiver_arg(nextX, lattice)

            if isinstance(f_sample_tran, lorenz_transformation) and my_encoder_cell is None:
                lattice = tf.placeholder(tf.float32, shape=(10, 10, 3, 3), name="lattice")
                nextX = SMC_train.get_nextX(lattice)
                mytrainer.set_quiver_arg(nextX, lattice)


        losses, tensors = mytrainer.train(obs_train, obs_test, print_freq, hidden_train, hidden_test)

    #pdb.set_trace()
    if generateTrainingData is False:
        losses, tensors = mytrainer.train(obs_train, obs_test, print_freq, x_0_init_mean, x_0_init_scale)
    # ======================================= another data saving part ======================================= #
    # _, R_square_trains, R_square_tests = losses
    if store_res:
        log_ZSMC_true_val, log_ZSMC_trains, log_ZSMC_tests, \
            MSE_trains, MSE_tests, \
            R_square_trains, R_square_tests = losses
        #log_true, log_train, ys_hat = tensors
        log_train, ys_hat = tensors

        Xs = log_train[0]
        Xs_val = mytrainer.evaluate(Xs, {obs: obs_train[0:saving_num],
                                         x_0: hidden_train[0:saving_num, 0],
                                         hidden: hidden_train[0:saving_num]})

        inferredX = np.average(Xs_val, axis=2)
        ys_hat_val = mytrainer.evaluate(ys_hat, {obs: obs_train[0:saving_num],
                                                 hidden: inferredX[0:saving_num]})

        print("finish evaluating training results")

        plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num=saving_num)
        plot_learning_results(RLT_DIR, Xs_val, hidden_train, saving_num=saving_num)
        plot_y_hat(RLT_DIR, ys_hat_val, obs_train, saving_num=saving_num)

        if isinstance(f_sample_tran, fhn_transformation):
            plot_fhn_results(RLT_DIR, Xs_val)

        if isinstance(f_sample_tran, lorenz_transformation):
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
                       "q_use_true_X": q_use_true_X}
        loss_dict = {"log_ZSMC_true": log_ZSMC_true_val,
                     "log_ZSMC_trains": log_ZSMC_trains,
                     "log_ZSMC_tests": log_ZSMC_tests,
                     #"MSE_true": MSE_true,
                     "MSE_trains": MSE_trains,
                     "MSE_tests": MSE_tests,
                     #"R_square_true": R_square_true,
                     "R_square_trains": R_square_trains,
                     "R_square_tests": R_square_tests}
        data_dict = {"params": params_dict,
                     "true_model_dict": true_model_dict,
                     "init_dict": init_dict,
                     "loss": loss_dict}

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
        if generateTrainingData:
            plot_log_ZSMC(RLT_DIR, log_ZSMC_true_val, log_ZSMC_trains, log_ZSMC_tests, print_freq)

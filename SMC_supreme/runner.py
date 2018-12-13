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

from sampler import create_dataset
from SMC import SMC


def main(_):
    FLAGS = tf.app.flags.FLAGS
    # print(FLAGS)

    # ============================================ parameter part ============================================ #
    # training hyperparameters
    Dx = FLAGS.Dx
    Dy = FLAGS.Dy
    n_particles = FLAGS.n_particles

    batch_size = FLAGS.batch_size
    lr = FLAGS.lr
    epoch = FLAGS.epoch
    seed = FLAGS.seed

    # --------------------- data set parameters --------------------- #
    # generate synthetic data?
    generateTrainingData = FLAGS.generateTrainingData

    # if reading data from file
    datadir = FLAGS.datadir
    datadict = FLAGS.datadict
    isPython2 = FLAGS.isPython2

    # time, n_train and n_test will be overwritten if loading data from the file
    time = FLAGS.time
    n_train = FLAGS.n_train
    n_test = FLAGS.n_test

    # --------------------- model parameters --------------------- #
    # network architectures
    q_train_layers = [int(x) for x in FLAGS.q_train_layers.split(",")]
    f_train_layers = [int(x) for x in FLAGS.f_train_layers.split(",")]
    g_train_layers = [int(x) for x in FLAGS.g_train_layers.split(",")]

    q_sigma_init, q_sigma_min = FLAGS.q_sigma_init, FLAGS.q_sigma_min
    f_sigma_init, f_sigma_min = FLAGS.f_sigma_init, FLAGS.f_sigma_min
    g_sigma_init, g_sigma_min = FLAGS.f_sigma_init, FLAGS.g_sigma_min

    # do q and f use the same network?
    use_bootstrap = FLAGS.use_bootstrap

    # if q takes y_t as input
    # if use_bootstrap, q_takes_y will be overwritten as False
    q_takes_y = FLAGS.q_takes_y

    # should q use true_X to sample? (useful for debugging)
    q_uses_true_X = FLAGS.q_uses_true_X

    # term to weight the added contribution of the MSE to the cost
    loss_beta = FLAGS.loss_beta

    # stop training early if validation set does not improve
    maxNumberNoImprovement = FLAGS.maxNumberNoImprovement

    # if x0 is learnable
    x_0_learnable = FLAGS.x_0_learnable

    # filtering or smoothing
    smoothing = FLAGS.smoothing

    # if f and q use residual
    use_residual = FLAGS.use_residual

    # if q, f and g networks also output covariance (sigma)
    output_cov = FLAGS.output_cov

    # --------------------- printing and data saving params --------------------- #
    print_freq = FLAGS.print_freq

    store_res = FLAGS.store_res
    rslt_dir_name = FLAGS.rslt_dir_name
    MSE_steps = FLAGS.MSE_steps

    # how many trajectories to draw in quiver plot
    quiver_traj_num = FLAGS.quiver_traj_num
    lattice_shape = [int(x) for x in FLAGS.lattice_shape.split(",")]

    saving_num = FLAGS.saving_num

    save_tensorboard = FLAGS.save_tensorboard
    save_model = FLAGS.save_model
    save_freq = FLAGS.save_freq

    MSE_steps = min(MSE_steps, time - 1)
    quiver_traj_num = min(quiver_traj_num, n_train, n_test)
    saving_num = min(saving_num, n_train, n_test)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    assert len(lattice_shape) == Dx
    lattice_shape.append(Dx)

    # ============================================= dataset part ============================================= #
    if generateTrainingData:

        # integrate differential equations to simulate the FHN or Lorenz systems
        sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
        f_params = (sigma, rho, beta, dt)

        # a, b, c, I, dt = 1.0, 0.95, 0.05, 1.0, 0.15
        # f_params = (a, b, c, I, dt)

        f_sample_cov = 0.0 * np.eye(Dx)

        # g_params = np.random.randn(Dy, Dx)  # np.array([[1.0, 1.0]]) or np.random.randn(Dy, Dx)
        g_params = np.array([[1.0, 0.0, 0.0]])
        g_sample_cov = 0.4 * np.eye(Dy)

        # f_sample_tran = fhn_transformation(f_params)
        f_sample_tran = lorenz_transformation(f_params)
        f_sample_dist = dirac_delta(f_sample_tran)

        g_sample_tran = linear_transformation(g_params)
        g_sample_dist = mvn(g_sample_tran, g_sample_cov)

        true_model_dict = {"f_params": f_params,
                           "f_cov": f_sample_cov,
                           "g_params": g_params,
                           "g_cov": g_sample_cov}

        # Create train and test dataset
        hidden_train, obs_train, hidden_test, obs_test = \
            create_dataset(n_train, n_test, time, Dx, Dy, f_sample_dist, g_sample_dist, lb=-2.5, ub=2.5)
        print("finished creating dataset")

    else:
        # load data
        with open(datadir + datadict, "rb") as handle:
            if isPython2:
                data = pickle.load(handle, encoding="latin1")
            else:
                data = pickle.load(handle)

        obs_train = data["Ytrain"]
        obs_test = data["Yvalid"]

        n_train = obs_train.shape[0]
        n_test = obs_test.shape[0]
        time = obs_train.shape[1]

        hidden_train = data["Xtrue"][:n_train]
        hidden_test = data["Xtrue"][n_train:]

        # reliminate quiver_traj_num and saving_num to avoid they > n_train or n_test
        quiver_traj_num = min(quiver_traj_num, n_train, n_test)
        saving_num = min(saving_num, n_train, n_test)
        MSE_steps = min(MSE_steps, time - 1)

        print("finished loading dataset")

    # ============================================== model part ============================================== #
    # placeholders
    if x_0_learnable:
        x_0 = tf.placeholder(tf.int32, shape=(batch_size), name="x_0")
        all_x_0 = tf.Variable(np.zeros((n_train + n_test, Dx)), dtype=tf.float32, name="all_x_0")
        x_0_val = tf.gather(all_x_0, x_0, name="x_0_val")
    else:
        x_0 = tf.placeholder(tf.float32, shape=(batch_size, Dx), name="x_0")
        x_0_val = tf.identity(x_0, name="x_0_val")
    obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name="obs")
    hidden = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name="hidden")

    # transformations
    # f_train_tran = MLP_transformation(f_train_layers, Dx, name="f_train_tran")
    q_train_tran = MLP_transformation(q_train_layers, Dx,
                                      use_residual=use_residual and not q_takes_y,
                                      output_cov=output_cov,
                                      name="q_train_tran")
    g_train_tran = MLP_transformation(g_train_layers, Dy,
                                      use_residual=False,
                                      output_cov=output_cov,
                                      name="g_train_tran")
    if use_bootstrap:
        f_train_tran = q_train_tran
        q_takes_y = False
    else:
        f_train_tran = MLP_transformation(f_train_layers, Dx,
                                          use_residual=use_residual,
                                          output_cov=output_cov,
                                          name="f_train_tran")

    q_train_dist = tf_mvn(q_train_tran, x_0_val, sigma_init=q_sigma_init, sigma_min=q_sigma_min, name="q_train_dist")
    g_train_dist = tf_mvn(g_train_tran, None, sigma_init=g_sigma_init, sigma_min=g_sigma_min, name="g_train_dist")
    if use_bootstrap:
        f_train_dist = q_train_dist
    else:
        f_train_dist = tf_mvn(f_train_tran, x_0_val, sigma_init=f_sigma_init, sigma_min=f_sigma_min, name="f_train_dist")

    init_dict = {"q_sigma_init": q_sigma_init,
                 "q_sigma_min": q_sigma_min,
                 "f_sigma_init": f_sigma_init,
                 "f_sigma_min": f_sigma_min,
                 "g_sigma_init": g_sigma_init,
                 "g_sigma_min": g_sigma_min}

    SMC_train = SMC(q_train_dist, f_train_dist, g_train_dist,
                    n_particles,
                    smoothing=smoothing,
                    q_takes_y=q_takes_y,
                    q_uses_true_X=q_uses_true_X,
                    name="log_ZSMC_train")

    # =========================================== data saving part =========================================== #
    if store_res:
        Experiment_params = {"np": n_particles,
                             "t": time,
                             "bs": batch_size,
                             "lr": lr,
                             "epoch": epoch,
                             "seed": seed,
                             "rslt_dir_name": rslt_dir_name}

        params_dict = {}
        params_list = sorted([param for param in dir(FLAGS) if param
                              not in ['h', 'help', 'helpfull', 'helpshort']])

        print("Experiment_params:")
        for param in params_list:
            params_dict[param] = str(getattr(FLAGS, param))
            print("\t" + param + ": " + str(getattr(FLAGS, param)))

        RLT_DIR = create_RLT_DIR(Experiment_params)
        print("RLT_DIR:", RLT_DIR)

    # ============================================= training part ============================================ #
    mytrainer = trainer(Dx, Dy,
                        n_particles, time,
                        batch_size, lr, epoch,
                        MSE_steps,
                        loss_beta,
                        maxNumberNoImprovement,
                        x_0_learnable)

    mytrainer.set_SMC(SMC_train)
    mytrainer.set_placeholders(x_0, obs, hidden)

    if store_res:
        mytrainer.set_rslt_saving(RLT_DIR, save_freq, saving_num, save_tensorboard, save_model)
        if Dx == 2 or Dx == 3:
            lattice = tf.placeholder(tf.float32, shape=lattice_shape, name="lattice")
            nextX = SMC_train.get_nextX(lattice)
            mytrainer.set_quiver_arg(nextX, lattice, quiver_traj_num, lattice_shape)

    losses, tensors = mytrainer.train(obs_train, obs_test, print_freq, hidden_train, hidden_test)

    # ======================================= another data saving part ======================================= #
    # _, R_square_trains, R_square_tests = losses
    if store_res:
        log_ZSMC_trains, log_ZSMC_tests, \
            MSE_trains, MSE_tests, \
            R_square_trains, R_square_tests = losses
        log_train, ys_hat = tensors

        Xs = log_train[0]
        if x_0_learnable:
            x_0_feed = n_train + np.arange(saving_num)
        else:
            x_0_feed = hidden_test[0:saving_num, 0]

        Xs_val = mytrainer.evaluate(Xs, {obs: obs_test[0:saving_num],
                                         x_0: x_0_feed,
                                         hidden: hidden_test[0:saving_num]})
        ys_hat_val = mytrainer.evaluate(ys_hat, {obs: obs_test[0:saving_num],
                                                 x_0: x_0_feed,
                                                 hidden: hidden_test[0:saving_num]})

        print("finish evaluating training results")

        plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num=saving_num)
        plot_learning_results(RLT_DIR, Xs_val, hidden_test, saving_num=saving_num)
        plot_y_hat(RLT_DIR, ys_hat_val, obs_test, saving_num=saving_num)

        if Dx == 2:
            plot_fhn_results(RLT_DIR, Xs_val)

        if Dx == 3:
            plot_lorenz_results(RLT_DIR, Xs_val)

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

        testing_data_dict = {"hidden_test": hidden_test[0:saving_num],
                             "obs_test": obs_test[0:saving_num]}
        learned_model_dict = {"Xs_val": Xs_val,
                              "ys_hat_val": ys_hat_val}
        data_dict["testing_data_dict"] = testing_data_dict
        data_dict["learned_model_dict"] = learned_model_dict

        with open(RLT_DIR + "data.p", "wb") as f:
            pickle.dump(data_dict, f)

        plot_MSEs(RLT_DIR, MSE_trains, MSE_tests, print_freq)
        plot_R_square(RLT_DIR, R_square_trains, R_square_tests, print_freq)
        plot_log_ZSMC(RLT_DIR, log_ZSMC_trains, log_ZSMC_tests, print_freq)

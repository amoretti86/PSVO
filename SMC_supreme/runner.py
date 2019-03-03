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
from attention import AttentionStack


def main(_):
    FLAGS = tf.app.flags.FLAGS
    # print(FLAGS)

    # ============================================ parameter part begins ============================================ #
    # training hyperparameters
    Dx = FLAGS.Dx
    Dy = FLAGS.Dy
    Di = FLAGS.Di
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
    # Feed-Forward Network (FFN) architectures
    q0_layers = [int(x) for x in FLAGS.q0_layers.split(",")]
    q1_layers = [int(x) for x in FLAGS.q1_layers.split(",")]
    q2_layers = [int(x) for x in FLAGS.q2_layers.split(",")]
    f_layers = [int(x) for x in FLAGS.f_layers.split(",")]
    g_layers = [int(x) for x in FLAGS.g_layers.split(",")]

    q0_sigma_init, q0_sigma_min = FLAGS.q0_sigma_init, FLAGS.q0_sigma_min
    q1_sigma_init, q1_sigma_min = FLAGS.q1_sigma_init, FLAGS.q1_sigma_min
    q2_sigma_init, q2_sigma_min = FLAGS.q2_sigma_init, FLAGS.q2_sigma_min
    f_sigma_init, f_sigma_min = FLAGS.f_sigma_init, FLAGS.f_sigma_min
    g_sigma_init, g_sigma_min = FLAGS.f_sigma_init, FLAGS.g_sigma_min

    # bidirectional RNN
    y_smoother_Dhs = [int(x) for x in FLAGS.y_smoother_Dhs.split(",")]
    X0_smoother_Dhs = [int(x) for x in FLAGS.X0_smoother_Dhs.split(",")]

    # Self-Attention encoder
    num_hidden_layers = FLAGS.num_hidden_layers
    num_heads = FLAGS.num_heads
    hidden_size = FLAGS.hidden_size
    filter_size = FLAGS.filter_size
    dropout_rate = FLAGS.dropout_rate

    # --------------------- FFN flags --------------------- #
    # do q and f use the same network?
    use_bootstrap = FLAGS.use_bootstrap

    # should q use true_X to sample? (useful for debugging)
    q_uses_true_X = FLAGS.q_uses_true_X

    # if f and q use residual
    use_residual = FLAGS.use_residual

    # if q, f and g networks also output covariance (sigma)
    output_cov = FLAGS.output_cov

    # if the networks only output diagonal value of cov matrix
    diag_cov = FLAGS.diag_cov

    # if q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t)
    # if True, use_bootstrap will be overwritten as True
    #          q_takes_y as False
    #          q_uses_true_X as False
    use_2_q = FLAGS.use_2_q

    # whether use input in q and f
    use_input = FLAGS.use_input

    # --------------------- FFBS flags --------------------- #

    # filtering or smoothing
    FFBS = FLAGS.FFBS

    # how fast the model transfers from filtering to smoothing
    smoothing_perc_factor = FLAGS.smoothing_perc_factor

    # whether use smoothing for inference or leaning
    FFBS_to_learn = FLAGS.FFBS_to_learn

    # --------------------- smoother flags --------------------- #

    # whether smooth observations with birdectional RNNs (bRNN) or self-attention encoders
    smooth_obs = FLAGS.smooth_obs

    # whether use bRNN or self-attention encoders to get X0 and encode observation
    use_RNN = FLAGS.use_RNN

    # whether use a separate RNN for getting X0
    X0_use_separate_RNN = FLAGS.X0_use_separate_RNN

    # whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn or tf.nn.bidirectional_dynamic_rnn
    # check https://stackoverflow.com/a/50552539 for differences between them
    use_stack_rnn = FLAGS.use_stack_rnn

    # --------------------- training flags --------------------- #

    # stop training early if validation set does not improve
    early_stop_patience = FLAGS.early_stop_patience

    # reduce learning rate when testing loss doesn't improve for some time
    lr_reduce_patience = FLAGS.lr_reduce_patience

    # the factor to reduce lr, new_lr = old_lr * lr_reduce_factor
    lr_reduce_factor = FLAGS.lr_reduce_factor

    # minimum lr
    min_lr = FLAGS.min_lr

    # whether use tf.stop_gradient when resampling and reweighting weights (during smoothing)
    use_stop_gradient = FLAGS.use_stop_gradient

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

    # ============================================ parameter part ends ============================================ #

    if use_2_q:
        q_uses_true_X = False

    MSE_steps = min(MSE_steps, time - 1)
    quiver_traj_num = min(quiver_traj_num, n_train, n_test)
    saving_num = min(saving_num, n_train, n_test)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    if Dx == 2:
        assert len(lattice_shape) == Dx
    lattice_shape.append(Dx)

    # ============================================= dataset part ============================================= #
    # generate data from simulation
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
        input_train = np.zeros((n_train, time, Di))
        input_test = np.zeros((n_test, time, Di))
        print("finished creating dataset")

    # load data from file
    else:
        with open(datadir + datadict, "rb") as handle:
            if isPython2:
                data = pickle.load(handle, encoding="latin1")
            else:
                data = pickle.load(handle)

        obs_train = data["Ytrain"]
        if "Ytest" in data:
            obs_test = data["Ytest"]
        else:
            obs_test = data["Yvalid"]

        if len(obs_train.shape) == 2:
            obs_train = np.expand_dims(obs_train, axis=2)
            obs_test = np.expand_dims(obs_test, axis=2)

        n_train = obs_train.shape[0]
        n_test = obs_test.shape[0]
        time = obs_train.shape[1]

        if not q_uses_true_X:
            hidden_train = np.zeros((n_train, time, Dx))
            hidden_test = np.zeros((n_test, time, Dx))
        else:
            if "Xtrue" in data:
                hidden_train = data["Xtrue"][:n_train]
                hidden_test = data["Xtrue"][n_train:]
            elif "Xtrain" in data and "Xtest" in data:
                hidden_train = data["Xtrain"]
                hidden_test = data["Xtest"]
            else:
                raise ValueError("Cannot find the keys for hidden_train and hidden_test in the data file")

        if use_input and "Itrain" in data and "Itest" in data:
            input_train = data["Itrain"]
            input_test = data["Itest"]
        else:
            input_train = np.zeros((n_train, time, Di))
            input_test = np.zeros((n_test, time, Di))

        # reliminate quiver_traj_num and saving_num to avoid they > n_train or n_test
        quiver_traj_num = min(quiver_traj_num, n_train, n_test)
        saving_num = min(saving_num, n_train, n_test)
        MSE_steps = min(MSE_steps, time - 1)

        print("finished loading dataset")

    # ============================================== model part ============================================== #
    # placeholders
    obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name="obs")
    hidden = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name="hidden")
    Input = tf.placeholder(tf.float32, shape=(batch_size, time, Di), name="Input")
    dropout = tf.placeholder(tf.float32, name="dropout")
    smoothing_perc = tf.placeholder(tf.float32, name="smoothing_perc")

    # transformations
    q0_tran = MLP_transformation(q0_layers, Dx,
                                 use_residual=False,
                                 output_cov=output_cov,
                                 diag_cov=diag_cov,
                                 name="q0_tran")
    q1_tran = MLP_transformation(q1_layers, Dx,
                                 use_residual=use_residual,
                                 output_cov=output_cov,
                                 diag_cov=diag_cov,
                                 name="q1_tran")
    if use_2_q:
        q2_tran = MLP_transformation(q2_layers, Dx,
                                     use_residual=False,
                                     output_cov=output_cov,
                                     diag_cov=diag_cov,
                                     name="q2_tran")
    else:
        q2_tran = None

    if use_bootstrap:
        f_tran = q1_tran
    else:
        f_tran = MLP_transformation(f_layers, Dx,
                                    use_residual=use_residual,
                                    output_cov=output_cov,
                                    diag_cov=diag_cov,
                                    name="f_tran")

    g_tran = MLP_transformation(g_layers, Dy,
                                use_residual=False,
                                output_cov=output_cov,
                                diag_cov=diag_cov,
                                name="g_tran")

    # distributions
    q0_dist = tf_mvn(q0_tran, sigma_init=q0_sigma_init, sigma_min=q0_sigma_min, name="q0_dist")
    q1_dist = tf_mvn(q1_tran, sigma_init=q1_sigma_init, sigma_min=q1_sigma_min, name="q1_dist")

    if use_2_q:
        q2_dist = tf_mvn(q2_tran, sigma_init=q2_sigma_init, sigma_min=q2_sigma_min, name="q2_dist")
    else:
        q2_dist = None

    if use_bootstrap:
        f_dist = q1_dist
    else:
        f_dist = tf_mvn(f_tran, sigma_init=f_sigma_init, sigma_min=f_sigma_min, name="f_dist")

    g_dist = tf_mvn(g_tran, sigma_init=g_sigma_init, sigma_min=g_sigma_min, name="g_dist")

    # smoothers
    if smooth_obs:
        if use_RNN:
            y_smoother_f = [tf.contrib.rnn.LSTMBlockCell(Dh, name="y_smoother_f_{}".format(i))
                            for i, Dh in enumerate(y_smoother_Dhs)]
            y_smoother_b = [tf.contrib.rnn.LSTMBlockCell(Dh, name="y_smoother_b_{}".format(i))
                            for i, Dh in enumerate(y_smoother_Dhs)]
            if X0_use_separate_RNN:
                X0_smoother_f = [tf.contrib.rnn.LSTMBlockCell(Dh, name="X0_smoother_f_{}".format(i))
                                 for i, Dh in enumerate(X0_smoother_Dhs)]
                X0_smoother_b = [tf.contrib.rnn.LSTMBlockCell(Dh, name="X0_smoother_b_{}".format(i))
                                 for i, Dh in enumerate(X0_smoother_Dhs)]
            else:
                X0_smoother_f = X0_smoother_b = None
            bRNN = (y_smoother_f, y_smoother_b, X0_smoother_f, X0_smoother_b)
            attention_encoder = None
        else:
            assert hidden_size % (num_heads * Dy) == 0
            attention_encoder = AttentionStack(num_hidden_layers, hidden_size, num_heads, filter_size, dropout)
            bRNN = None
    else:
        bRNN = attention_encoder = None

    # SMC class to calculate loss
    SMC_train = SMC(q0_dist, q1_dist, q2_dist, f_dist, g_dist,
                    n_particles,
                    q_uses_true_X=q_uses_true_X,
                    bRNN=bRNN,
                    attention_encoder=attention_encoder,
                    X0_use_separate_RNN=X0_use_separate_RNN,
                    use_stack_rnn=use_stack_rnn,
                    FFBS=FFBS,
                    smoothing_perc=smoothing_perc,
                    FFBS_to_learn=FFBS_to_learn,
                    use_stop_gradient=use_stop_gradient,
                    use_input=use_input,
                    name="log_ZSMC_train")

    # =========================================== data saving part =========================================== #
    # create dir to store results
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

        with open(RLT_DIR + "param.json", "w") as f:
            json.dump(params_dict, f, indent=4, cls=NumpyEncoder)

    # ============================================= training part ============================================ #
    mytrainer = trainer(Dx, Dy,
                        n_particles, time,
                        batch_size, lr, epoch,
                        MSE_steps,
                        smoothing_perc_factor)

    mytrainer.training_params(early_stop_patience, lr_reduce_factor, lr_reduce_patience, min_lr, dropout_rate)
    mytrainer.set_SMC(SMC_train)
    mytrainer.set_placeholders(obs, hidden, Input, dropout, smoothing_perc)

    if store_res:
        mytrainer.set_rslt_saving(RLT_DIR, save_freq, saving_num, save_tensorboard, save_model)
        if Dx == 2:
            lattice = tf.placeholder(tf.float32, shape=lattice_shape, name="lattice")
            nextX = SMC_train.get_nextX(lattice)
            mytrainer.set_quiver_arg(nextX, lattice, quiver_traj_num, lattice_shape)
        elif Dx == 3:
            mytrainer.draw_quiver_during_training = True

    history, log = mytrainer.train(obs_train, obs_test,
                                   hidden_train, hidden_test,
                                   input_train, input_test,
                                   print_freq)

    # ======================================= another data saving part ======================================= #
    if store_res:
        Xs, y_hat = log["Xs"], log["y_hat"]

        Xs_val = mytrainer.evaluate(Xs, {obs: obs_test[0:saving_num],
                                         hidden: hidden_test[0:saving_num],
                                         Input: input_test[0:saving_num],
                                         dropout: np.zeros(saving_num),
                                         smoothing_perc: np.ones(saving_num)})
        y_hat_val = mytrainer.evaluate(y_hat, {obs: obs_test[0:saving_num],
                                               hidden: hidden_test[0:saving_num],
                                               Input: input_test[0:saving_num],
                                               dropout: np.zeros(saving_num),
                                               smoothing_perc: np.ones(saving_num)})

        print("finish evaluating training results")

        plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num=saving_num)
        plot_learning_results(RLT_DIR, Xs_val, hidden_test, saving_num=saving_num)
        plot_y_hat(RLT_DIR, y_hat_val, obs_test, saving_num=saving_num)

        if Dx == 2:
            plot_fhn_results(RLT_DIR, Xs_val)

        if Dx == 3:
            plot_lorenz_results(RLT_DIR, Xs_val)

        data_dict = {"history": history}

        if generateTrainingData:
            data_dict["true_model_dict"] = true_model_dict

        with open(RLT_DIR + "data.json", "w") as f:
            json.dump(data_dict, f, indent=4, cls=NumpyEncoder)

        testing_data_dict = {"hidden_test": hidden_test[0:saving_num],
                             "obs_test": obs_test[0:saving_num]}
        learned_model_dict = {"Xs_val": Xs_val,
                              "y_hat_val": y_hat_val}
        data_dict["testing_data_dict"] = testing_data_dict
        data_dict["learned_model_dict"] = learned_model_dict

        with open(RLT_DIR + "data.p", "wb") as f:
            pickle.dump(data_dict, f)

        plot_MSEs(RLT_DIR, history["MSE_trains"], history["MSE_tests"], print_freq)
        plot_R_square(RLT_DIR, history["R_square_trains"], history["R_square_tests"], print_freq)
        plot_log_ZSMC(RLT_DIR, history["log_ZSMC_trains"], history["log_ZSMC_tests"], print_freq)

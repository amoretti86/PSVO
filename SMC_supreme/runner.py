import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

# for data saving stuff
import pickle
import json
import os
import pdb

# import from files
from SMC_supreme.model import SSM
from SMC_supreme.SMC import SMC
from SMC_supreme.trainer import trainer

from SMC_supreme.rslts_saving.rslts_saving import *
from SMC_supreme.rslts_saving.fhn_rslts_saving import *
from SMC_supreme.rslts_saving.lorenz_rslts_saving import *

from SMC_supreme.utils.data_generator import generate_dataset
from SMC_supreme.utils.data_loader import load_data


def main(_):
    FLAGS = tf.app.flags.FLAGS  # @UndefinedVariable

    # ========================================= parameter part begins ========================================== #
    Dx = FLAGS.Dx

    # --------------------- SSM flags --------------------- #
    # should q use true_X to sample? (useful for debugging)
    q_uses_true_X = FLAGS.q_uses_true_X

    # do q and f use the same network?
    use_bootstrap = FLAGS.use_bootstrap

    # if q uses two networks q1(x_t|x_t-1) and q2(x_t|y_t)
    # if True, use_bootstrap will be overwritten as True
    #          q_takes_y as False
    #          q_uses_true_X as False
    use_2_q = FLAGS.use_2_q

    # whether use input in q and f
    use_input = FLAGS.use_input

    # whether emission uses Poisson distribution
    poisson_emission = FLAGS.poisson_emission

    # whether transitions (q1 and f) use Normalizing Flow
    flow_transition = FLAGS.flow_transition

    # --------------------- FFN flags --------------------- #
    # if f and q use residual
    use_residual = FLAGS.use_residual

    # if q, f and g networks also output covariance (sigma)
    output_cov = FLAGS.output_cov

    # if the networks only output diagonal value of cov matrix
    diag_cov = FLAGS.diag_cov

    # -------------------------------- TFS flags -------------------------------- #

    # whether use Two Filter Smoothing
    TFS = FLAGS.TFS

    # whether backward filtering in TFS uses different q0
    TFS_use_diff_q0 = FLAGS.TFS_use_diff_q0

    # -------------------------------- FFBS flags ------------------------------- #

    # whether use Forward Filtering Backward Smoothing
    FFBS = FLAGS.FFBS

    # whether use FFBS for inference or leaning
    FFBS_to_learn = FLAGS.FFBS_to_learn

    # ------------------------------ smoother flags ----------------------------- #

    # whether smooth observations with birdectional RNNs (bRNN) or self-attention encoders
    smooth_obs = FLAGS.smooth_obs

    # whether use a separate RNN for getting X0
    X0_use_separate_RNN = FLAGS.X0_use_separate_RNN

    # whether use tf.contrib.rnn.stack_bidirectional_dynamic_rnn or tf.nn.bidirectional_dynamic_rnn
    # check https://stackoverflow.com/a/50552539 for differences between them
    use_stack_rnn = FLAGS.use_stack_rnn

    # --------------------- printing and data saving params --------------------- #

    print_freq = FLAGS.print_freq

    if FLAGS.use_2_q:
        FLAGS.q_uses_true_X = q_uses_true_X = False
    if FLAGS.flow_transition:
        FLAGS.use_input = use_input = False

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ============================================= dataset part ============================================= #
    # generate data from simulation
    if FLAGS.generateTrainingData:
        model = "lorenz"
        hidden_train, hidden_test, obs_train, obs_test, input_train, input_test = \
            generate_dataset(FLAGS.n_train, FLAGS.n_test, FLAGS.time, model=model, Dy=FLAGS.Dy, lb=-2.5, ub=2.5)

    # load data from file
    else:
        hidden_train, hidden_test, obs_train, obs_test, input_train, input_test = \
            load_data(FLAGS.datadir + FLAGS.datadict, Dx, FLAGS.Di, FLAGS.isPython2, use_input, q_uses_true_X)
        FLAGS.n_train, FLAGS.n_test, FLAGS.time = obs_train.shape[0], obs_test.shape[0], obs_test.shape[1]

    # clip saving_num to avoid it > n_train or n_test
    FLAGS.MSE_steps = min(FLAGS.MSE_steps, FLAGS.time - 1)
    FLAGS.saving_num = min(FLAGS.saving_num, FLAGS.n_train, FLAGS.n_test)
    saving_num = FLAGS.saving_num

    print("finished preparing dataset")

    # ============================================== model part ============================================== #
    SSM_model = SSM(FLAGS)
#     SSM_model = SSM(FLAGS,
#                     use_residual=use_residual,
#                     output_cov=output_cov,
#                     diag_cov=diag_cov,
#                     use_bootstrap=use_bootstrap,
#                     use_2_q=use_2_q,
#                     flow_transition=flow_transition,
#                     poisson_emission=poisson_emission,
#                     TFS=TFS,
#                     TFS_use_diff_q0=TFS_use_diff_q0,
#                     smooth_obs=smooth_obs,
#                     X0_use_separate_RNN=X0_use_separate_RNN,
#                     use_stack_rnn=use_stack_rnn)

    # SMC class to calculate loss
    SMC_train = SMC(SSM_model,
                    FLAGS.n_particles,
                    q_uses_true_X=q_uses_true_X,
                    use_input=use_input,
                    X0_use_separate_RNN=X0_use_separate_RNN,
                    use_stack_rnn=use_stack_rnn,
                    FFBS=FFBS,
                    FFBS_to_learn=FFBS_to_learn,
                    TFS=TFS,
                    name="log_ZSMC_train")

    # =========================================== data saving part =========================================== #
    # create dir to save results
    Experiment_params = {"np":            FLAGS.n_particles,
                         "t":             FLAGS.time,
                         "bs":            FLAGS.batch_size,
                         "lr":            FLAGS.lr,
                         "epoch":         FLAGS.epoch,
                         "seed":          FLAGS.seed,
                         "rslt_dir_name": FLAGS.rslt_dir_name}

    RLT_DIR = create_RLT_DIR(Experiment_params)
    save_experiment_param(RLT_DIR, FLAGS)
    print("RLT_DIR:", RLT_DIR)

    # ============================================= training part ============================================ #
    mytrainer = trainer(SSM_model, SMC_train, FLAGS)
    mytrainer.init_data_saving(RLT_DIR)

    history, log = mytrainer.train(obs_train, obs_test,
                                   hidden_train, hidden_test,
                                   input_train, input_test,
                                   print_freq)

    # ======================================== final data saving part ======================================== #
    with open(RLT_DIR + "history.json", "w") as f:
        json.dump(history, f, indent=4, cls=NumpyEncoder)

    Xs, y_hat = log["Xs"], log["y_hat"]
    Xs_val = mytrainer.evaluate(Xs, mytrainer.saving_feed_dict)
    y_hat_val = mytrainer.evaluate(y_hat, mytrainer.saving_feed_dict)
    print("finish evaluating training results")

    plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num=saving_num)
    plot_y_hat(RLT_DIR, y_hat_val, obs_test, saving_num=saving_num)

    if Dx == 2:
        plot_fhn_results(RLT_DIR, Xs_val)
    if Dx == 3:
        plot_lorenz_results(RLT_DIR, Xs_val)

    testing_data_dict = {"hidden_test": hidden_test[0:saving_num],
                         "obs_test": obs_test[0:saving_num]}
    learned_model_dict = {"Xs_val": Xs_val,
                          "y_hat_val": y_hat_val}
    data_dict = {"testing_data_dict": testing_data_dict,
                 "learned_model_dict": learned_model_dict}

    with open(RLT_DIR + "data.p", "wb") as f:
        pickle.dump(data_dict, f)

    plot_MSEs(RLT_DIR, history["MSE_trains"], history["MSE_tests"], print_freq)
    plot_R_square(RLT_DIR, history["R_square_trains"], history["R_square_tests"], print_freq)
    plot_log_ZSMC(RLT_DIR, history["log_ZSMC_trains"], history["log_ZSMC_tests"], print_freq)


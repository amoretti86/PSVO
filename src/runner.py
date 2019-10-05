import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

# for data saving stuff
import pickle
import json
import os
import pdb

# import from files
from model import SSM
from trainer import trainer

from SMC.SVO import SVO
from SMC.PSVO import PSVO
from SMC.PSVOwR import PSVOwR
from SMC.IWAE import IWAE
from SMC.AESMC import AESMC


from rslts_saving.rslts_saving import *
from rslts_saving.fhn_rslts_saving import *
from rslts_saving.lorenz_rslts_saving import *

from utils.data_generator import generate_dataset
from utils.data_loader import load_data


def main(_):
    FLAGS = tf.app.flags.FLAGS

    # ========================================= parameter part begins ========================================== #
    Dx = FLAGS.Dx
    print_freq = FLAGS.print_freq

    if FLAGS.use_2_q:
        FLAGS.q_uses_true_X = False

    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ============================================= dataset part ============================================= #
    # generate data from simulation
    if FLAGS.generateTrainingData:
        model = "fhn"
        hidden_train, hidden_test, obs_train, obs_test  = \
            generate_dataset(FLAGS.n_train, FLAGS.n_test, FLAGS.time, model=model, Dy=FLAGS.Dy, lb=-2.5, ub=2.5)

    # load data from file
    else:
        hidden_train, hidden_test, obs_train, obs_test = \
            load_data(FLAGS.datadir + FLAGS.datadict, Dx, FLAGS.isPython2, FLAGS.q_uses_true_X)
        FLAGS.n_train, FLAGS.n_test, FLAGS.time = obs_train.shape[0], obs_test.shape[0], obs_test.shape[1]

    # clip saving_num to avoid it > n_train or n_test
    FLAGS.MSE_steps  = min(FLAGS.MSE_steps, FLAGS.time - 1)
    FLAGS.saving_num = saving_num = min(FLAGS.saving_num, FLAGS.n_train, FLAGS.n_test)

    print("finished preparing dataset")

    # ============================================== model part ============================================== #
    SSM_model = SSM(FLAGS)

    # at most one of them can be set to True
    assert FLAGS.PSVO + FLAGS.SVO + FLAGS.AESMC + FLAGS.IWAE < 2

    # SMC class to calculate loss
    if FLAGS.PSVO:
        SMC_train = PSVO(SSM_model, FLAGS)
    elif FLAGS.PSVOwR:
        SMC_train = PSVOwR(SSM_model, FLAGS)
    elif FLAGS.SVO:
        SMC_train = SVO(SSM_model, FLAGS)
    elif FLAGS.AESMC:
        SMC_train = AESMC(SSM_model, FLAGS)
    elif FLAGS.IWAE:
        SMC_train = IWAE(SSM_model, FLAGS)
    else:
        raise ValueError("Choose one of objectives among: PSVO, SVO, AESMC, IWAE")

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

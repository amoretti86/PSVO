import json
import numpy as np
from datetools import addDateTime
import os

import seaborn as sns
import matplotlib.pyplot as plt


def create_RLT_DIR(Experiment_params):
    # create the dir to save data
    # Experiment_params is a dict containing param_name&param pair
    # Experiment_params must contain "rslt_dir_name":rslt_dir_name
    cur_date = addDateTime()

    local_rlt_root = 'rslts/' + Experiment_params['rslt_dir_name'] + '/'

    params_str = ""
    for param_name, param in Experiment_params.items():
        if param_name == 'rslt_dir_name':
            continue
        params_str += param_name + '_' + str(param) + '_'

    RLT_DIR = os.path.join(os.getcwd(), local_rlt_root + params_str + cur_date + '/')

    if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)

    return RLT_DIR

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def plot_training_data(RLT_DIR, hidden_train, obs_train):
    # Plot and save training data
    if not os.path.exists(RLT_DIR+"/Training Data"): os.makedirs(RLT_DIR+"/Training Data")
    for i in range(len(hidden_train)):
        plt.figure()
        plt.title("Training Time Series")
        plt.xlabel("Time")
        plt.plot(hidden_train[i], c='red')
        plt.plot(obs_train[i], c='blue')
        sns.despine()
        plt.savefig(RLT_DIR+"Training Data/{}".format(i))
        plt.close()

def plot_learning_results(RLT_DIR, Xs_val, hidden_train):
    # Plot and save learning results
    if not os.path.exists(RLT_DIR+"/Learning Results"): os.makedirs(RLT_DIR+"/Learning Results")
    n_train = len(hidden_train)
    for i in range(n_train):
        plt.figure()
        plt.title("hidden state 0")
        plt.xlabel("Time")
        plt.plot(np.average(Xs_val[i, :, :, 0], axis = 1), alpha = 0.5, c = 'black')
        plt.plot(hidden_train[i][:, 0], c='yellow')
        sns.despine()
        plt.savefig(RLT_DIR+"/Learning Results/h_0_{}".format(i))
        plt.close()

        plt.figure()
        plt.title("hidden state 1")
        plt.xlabel("Time")
        plt.plot(np.average(Xs_val[i, :, :, 1], axis = 1), alpha = 0.5, c = 'black')
        plt.plot(hidden_train[i][:, 1], c='yellow')
        sns.despine()
        plt.savefig(RLT_DIR+"/Learning Results/h_1_{}".format(i))
        plt.close()

def plot_losses(RLT_DIR, true_log_ZSMC_val, log_ZSMC_trains, log_ZSMC_tests):
    # Plot and save losses
    plt.figure()
    plt.plot([true_log_ZSMC_val] * len(log_ZSMC_trains))
    plt.plot(log_ZSMC_trains)
    plt.plot(log_ZSMC_tests)
    plt.legend(['true_log_ZSMC_val', 'log_ZSMC_trains', 'log_ZSMC_tests'])
    sns.despine()
    plt.savefig(RLT_DIR + "log_ZSMC")
    plt.show()

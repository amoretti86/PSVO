import json
import numpy as np
from datetools import addDateTime
import os

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        params_str += '_' + param_name + '_' + str(param)

    RLT_DIR = os.path.join(os.getcwd(), local_rlt_root + cur_date + params_str  + '/')
    MODEL_DIR = os.path.join(RLT_DIR, 'model/')

    if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

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

def plot_training_data(RLT_DIR, hidden_train, obs_train, max_fig_num = 20):
    # Plot and save training data
    if not os.path.exists(RLT_DIR+"/Training Data"): os.makedirs(RLT_DIR+"/Training Data")
    for i in range(min(len(hidden_train), max_fig_num)):
        plt.figure()
        plt.title("Training Time Series")
        plt.xlabel("Time")
        plt.plot(hidden_train[i], c='red')
        plt.plot(obs_train[i], c='blue')
        sns.despine()
        plt.savefig(RLT_DIR+"Training Data/{}".format(i))
        plt.close()

def plot_learning_results(RLT_DIR, Xs_val, hidden_train, max_fig_num = 20):
    # Plot and save learning results
    if not os.path.exists(RLT_DIR+"/Learning Results"): os.makedirs(RLT_DIR+"/Learning Results")
    n_train = min(len(hidden_train), max_fig_num)
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

def plot_MSEs(RLT_DIR, MSE_true, MSE_trains, MSE_tests):
    # Plot and save losses
    plt.figure()
    plt.plot([MSE_true] * len(MSE_trains))
    plt.plot(MSE_trains)
    plt.plot(MSE_tests)
    plt.legend(['MSE_trues', 'MSE_trains', 'MSE_tests'])
    sns.despine()
    plt.savefig(RLT_DIR + "MSE")
    plt.show()

def plot_fhn_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR+"/FHN 2D plots"): os.makedirs(RLT_DIR+"/FHN 2D plots")
    for i in range(Xs_val.shape[0]):
        plt.figure()
        plt.title("hidden state for all particles")
        plt.xlabel("x_dim 1")
        plt.ylabel("x_dim 2")
        for j in range(Xs_val.shape[2]):
            plt.plot(Xs_val[i, :, j, 0], Xs_val[i, :, j, 1])
        sns.despine()
        plt.savefig(RLT_DIR+"/FHN 2D plots/All_x_paths_{}".format(i))
        plt.close()

def plot_lorenz_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR+"/Lorenz 3D plots"): os.makedirs(RLT_DIR+"/Lorenz 3D plots")
    for i in range(Xs_val.shape[0]):        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.title("hidden state for all particles")
        ax.set_xlabel('x_dim 1')
        ax.set_ylabel('x_dim 2')
        ax.set_zlabel('x_dim 3')
        for j in range(Xs_val.shape[2]):
            ax.plot(Xs_val[i, :, j, 0], Xs_val[i, :, j, 1], Xs_val[i, :, j, 2])
        plt.savefig(RLT_DIR+"/Lorenz 3D plots/All_x_paths_{}".format(i))
        plt.close()

def plot_y_hat(RLT_DIR, ys_hat_val, obs):
    if not os.path.exists(RLT_DIR+"/y_hat plots"): os.makedirs(RLT_DIR+"/y_hat plots")
    for i in range(ys_hat_val.shape[0]):
        for j in range(ys_hat_val.shape[-1]):
            plt.figure()
            plt.title("obs dim {}".format(j))
            plt.xlabel("Time")
            plt.plot(obs[i][:, j])
            for k in range(ys_hat_val.shape[2]):
                plt.plot(range(k, k + ys_hat_val.shape[1]), ys_hat_val[i, :, k, j], '*-')
            sns.despine()
            plt.savefig(RLT_DIR+"/y_hat plots/obs_{}_{}".format(j, i))
            plt.close()

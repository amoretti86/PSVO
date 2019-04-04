import numpy as np

import os
import json
import pickle

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SMC_supreme.rslts_saving.datetools import addDateTime


def create_RLT_DIR(Experiment_params):
    # create the dir to save data
    # Experiment_params is a dict containing param_name&param pair
    # Experiment_params must contain "rslt_dir_name":rslt_dir_name
    cur_date = addDateTime()

    local_rlt_root = "rslts/" + Experiment_params["rslt_dir_name"] + "/"

    params_str = ""
    for param_name, param in Experiment_params.items():
        if param_name == "rslt_dir_name":
            continue
        params_str += "_" + param_name + "_" + str(param)

    RLT_DIR = os.getcwd().replace("\\", "/") + "/" + local_rlt_root + cur_date[1:] + params_str + "/"

    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)

    return RLT_DIR


def save_experiment_param(RLT_DIR, FLAGS):
    params_dict = {}
    params_list = sorted([param for param in dir(FLAGS) if param
                          not in ['h', 'help', 'helpfull', 'helpshort']])

    print("Experiment_params:")
    for param in params_list:
        params_dict[param] = str(getattr(FLAGS, param))
        print("\t" + param + ": " + str(getattr(FLAGS, param)))

    with open(RLT_DIR + "param.json", "w") as f:
        json.dump(params_dict, f, indent=4, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):

    # Special json encoder for numpy types
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                      np.int16, np.int32, np.int64, np.uint8,
                      np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                        np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num=20):
    # Plot and save training data
    if not os.path.exists(RLT_DIR + "Training Data"):
        os.makedirs(RLT_DIR + "Training Data")
    saving_num = min(len(hidden_train), saving_num)
    for i in range(saving_num):
        plt.figure()
        plt.title("Training Time Series")
        plt.xlabel("Time")
        plt.plot(hidden_train[i], c="red")
        plt.plot(obs_train[i], c="blue")
        sns.despine()
        plt.savefig(RLT_DIR + "Training Data/{}".format(i))
        plt.close()


def plot_learning_results(RLT_DIR, Xs_val, hidden_train, saving_num=20):
    # Plot and save learning results
    if not os.path.exists(RLT_DIR + "Learning Results"):
        os.makedirs(RLT_DIR + "Learning Results")
    saving_num = min(len(hidden_train), saving_num)
    for i in range(saving_num):
        for j in range(Xs_val.shape[-1]):
            plt.figure()
            plt.title("hidden state {}".format(j))
            plt.xlabel("Time")
            plt.plot(np.mean(Xs_val[i, :, :, j], axis=1), alpha=0.5, c="black")
            plt.plot(hidden_train[i, :, j], c="yellow")
            plt.legend(["prediction", "ground truth"])
            sns.despine()
            plt.savefig(RLT_DIR + "Learning Results/h_dim_{}_idx_{}".format(j, i))
            plt.close()


def plot_log_ZSMC(RLT_DIR, log_ZSMC_trains, log_ZSMC_tests, print_freq):
    epoch = np.arange(len(log_ZSMC_trains)) * print_freq
    plt.figure()
    plt.plot(epoch, log_ZSMC_trains)
    plt.plot(epoch, log_ZSMC_tests)
    plt.legend(["log_ZSMC_train", "log_ZSMC_test"])
    sns.despine()
    plt.savefig(RLT_DIR + "log_ZSMC")
    plt.show()


def plot_MSEs(RLT_DIR, MSE_trains, MSE_tests, print_freq):
    if not os.path.exists(RLT_DIR + "MSE"):
        os.makedirs(RLT_DIR + "MSE")
    # Plot and save losses
    plt.figure()
    for i in range(len(MSE_trains)):
        plt.plot(MSE_trains[i])
        plt.plot(MSE_tests[i])
        plt.xlabel("k")
        plt.legend(["MSE_train", "MSE_test"])
        sns.despine()
        plt.savefig(RLT_DIR + "MSE/epoch_{}".format(i * print_freq))
        plt.close()


def plot_R_square(RLT_DIR, R_square_trains, R_square_tests, print_freq):
    if not os.path.exists(RLT_DIR + "R_square"):
        os.makedirs(RLT_DIR + "R_square")
    # Plot and save losses
    plt.figure()
    for i in range(len(R_square_trains)):
        plt.plot(R_square_trains[i])
        plt.plot(R_square_tests[i])
        plt.ylim([0.0, 1.05])
        plt.xlabel("K")
        plt.legend(["Train $R^2_k$", "Test $R^2_k$"], loc='best')
        sns.despine()
        plt.savefig(RLT_DIR + "R_square/epoch_{}".format(i * print_freq))
        plt.close()


def plot_R_square_epoch(RLT_DIR, R_square_trains, R_square_tests, epoch):
    if not os.path.exists(RLT_DIR + "R_square"):
        os.makedirs(RLT_DIR + "R_square")
    plt.figure()
    plt.plot(R_square_trains)
    plt.plot(R_square_tests)
    plt.ylim([0.0, 1.05])
    plt.xlabel("K")
    plt.legend(["Train $R^2_k$", "Test $R^2_k$"], loc='best')
    sns.despine()
    plt.savefig(RLT_DIR + "R_square/epoch_{}".format(epoch))
    plt.close()


def plot_fhn_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR + "FHN 2D plots"):
        os.makedirs(RLT_DIR + "FHN 2D plots")
    for i in range(Xs_val.shape[0]):
        plt.figure()
        plt.title("hidden state for all particles")
        plt.xlabel("x_dim 1")
        plt.ylabel("x_dim 2")
        for j in range(Xs_val.shape[2]):
            plt.plot(Xs_val[i, :, j, 0], Xs_val[i, :, j, 1])
        sns.despine()
        plt.savefig(RLT_DIR + "/FHN 2D plots/All_x_paths_{}".format(i))
        plt.close()


def plot_lorenz_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR + "Lorenz 3D plots"):
        os.makedirs(RLT_DIR + "Lorenz 3D plots")
    for i in range(Xs_val.shape[0]):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        plt.title("hidden state for all particles")
        ax.set_xlabel("x_dim 1")
        ax.set_ylabel("x_dim 2")
        ax.set_zlabel("x_dim 3")
        for j in range(Xs_val.shape[2]):
            ax.plot(Xs_val[i, :, j, 0], Xs_val[i, :, j, 1], Xs_val[i, :, j, 2])
        plt.savefig(RLT_DIR + "/Lorenz 3D plots/All_x_paths_{}".format(i))
        plt.close()


def plot_y_hat(RLT_DIR, ys_hat_val, obs, saving_num=20):
    if not os.path.exists(RLT_DIR + "y_hat plots"):
        os.makedirs(RLT_DIR + "y_hat plots")

    _, time, Dy = obs.shape
    saving_num = min(len(ys_hat_val), saving_num)

    for i in range(saving_num):
        for j in range(Dy):
            plt.figure()
            plt.title("obs dim {}".format(j))
            plt.xlabel("Time")
            plt.plot(obs[i, :, j])
            for k, ys_k_hat_val in enumerate(ys_hat_val):
                plt.plot(range(k, time), ys_k_hat_val[i, :, j], "--")
            sns.despine()
            plt.savefig(RLT_DIR + "/y_hat plots/obs_dim_{}_idx_{}".format(j, i))
            plt.close()

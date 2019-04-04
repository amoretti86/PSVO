import numpy as np

# for data saving stuff
import pickle
import os

# import from files
from SMC_supreme.transformation.fhn import fhn_transformation, tf_fhn_transformation
from SMC_supreme.transformation.linear import linear_transformation, tf_linear_transformation
from SMC_supreme.transformation.lorenz import lorenz_transformation, tf_lorenz_transformation
from SMC_supreme.transformation.MLP import MLP_transformation

from SMC_supreme.distribution.dirac_delta import dirac_delta
from SMC_supreme.distribution.mvn import mvn, tf_mvn
from SMC_supreme.distribution.poisson import poisson, tf_poisson

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from SMC_supreme.rslts_saving.rslts_saving import *


def generate_hidden_obs(time, Dx, Dy, x_0, f, g, keep_first=True):
    """
    Generate hidden states and observation
    f: transition class with x_t = g.sample(x_t-1)
    g: emission class with y_t = g.sample(x_t)
    """
    X = np.zeros((time + 1, Dx))
    Y = np.zeros((time + 1, Dy))

    X[0] = x_0
    Y[0] = g.sample(x_0)

    for t in range(1, time + 1):
        X[t] = f.sample(X[t - 1])
        Y[t] = g.sample(X[t])

    if keep_first:
        return X[:-1], Y[:-1]
    else:
        return X[1:], Y[1:]


def create_dataset(n_train, n_test, time_list, Dx, Dy, f, g, f_params_list, x_0_in=None, lb=None, ub=None):
    time = np.sum(time_list)

    hidden_train, obs_train = np.zeros((n_train, time, Dx)), np.zeros((n_train, time, Dy))
    hidden_test, obs_test = np.zeros((n_test, time, Dx)), np.zeros((n_test, time, Dy))

    if x_0_in is None and (lb and ub) is None:
        assert False, 'must specify x_0 or (lb and ub)'

    x_0_in_copy = x_0_in

    for i in range(n_train + n_test):
        hidden_list = []
        obs_list = []

        x_0_in = x_0_in_copy
        for time_part, f_params in zip(time_list, f_params_list):
            f.transformation.params = f_params
            if x_0_in is None:
                x_0_in = np.random.uniform(low=lb, high=ub, size=Dx)

            keep_first = (len(hidden_list) == 0)
            hidden, obs = generate_hidden_obs(time_part, Dx, Dy, x_0_in, f, g, keep_first)
            x_0_in = hidden[-1]

            hidden_list.append(hidden)
            obs_list.append(obs)

        hidden = np.concatenate(hidden_list)
        obs = np.concatenate(obs_list)

        if i < n_train:
            hidden_train[i] = hidden
            obs_train[i] = obs
        else:
            hidden_test[i - n_train] = hidden
            obs_test[i - n_train] = obs

    return hidden_train, obs_train, hidden_test, obs_test


def plot_fhn_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR + "/FHN 2D plots"):
        os.makedirs(RLT_DIR + "/FHN 2D plots")
    for i in range(Xs_val.shape[0]):
        plt.figure()
        plt.title("hidden state for all particles")
        plt.xlabel("x_dim 1")
        plt.ylabel("x_dim 2")
        for j in range(Xs_val.shape[2]):
            plt.plot(Xs_val[i, :, 0], Xs_val[i, :, 1])
        sns.despine()
        plt.savefig(RLT_DIR + "/FHN 2D plots/All_x_paths_{}".format(i))
        plt.close()


def plot_lorenz_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR + "/Lorenz 3D plots"):
        os.makedirs(RLT_DIR + "/Lorenz 3D plots")
    for i in range(Xs_val.shape[0]):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        plt.title("hidden state for all particles")
        ax.set_xlabel("x_dim 1")
        ax.set_ylabel("x_dim 2")
        ax.set_zlabel("x_dim 3")
        for j in range(Xs_val.shape[2]):
            ax.plot(Xs_val[i, :, 0], Xs_val[i, :, 1], Xs_val[i, :, 2])
        plt.savefig(RLT_DIR + "/Lorenz 3D plots/All_x_paths_{}".format(i))
        plt.close()


if __name__ == "__main__":
    Dx = 2
    Dy = 2

    time_list = [200] * 1
    n_train = 200
    n_test = 40

    datadict = {}

    # integrate differential equations to simulate the FHN or Lorenz systems
    # sigma, rho, beta, dt = 10.0, 28.0, 8.0 / 3.0, 0.01
    # f_params = (sigma, rho, beta, dt)

    a, b, c, I, dt = 1.0, 0.95, 0.05, 1.0, 0.15
    f_params1 = (a, b, c, I, dt)

    # a, b, c, I, dt = 1.0, 0.95, 0.05, 3.0, 0.15
    # f_params2 = (a, b, c, I, dt)

    f_sample_cov = 0.0 * np.eye(Dx)

    f_params_list = [f_params1] * 1

    # g_params = np.random.randn(Dy, Dx)  # np.array([[1.0, 1.0]]) or np.random.randn(Dy, Dx)
    g_params = np.array([[1.0, 0.0], [0.0, 1.0]])
    g_sample_cov = 0.1 * np.eye(Dy)

    datadict["time_list"] = time_list
    datadict["a_b_c_I_dt_list"] = f_params_list
    datadict["g_mat"] = g_params
    datadict["g_cov"] = g_sample_cov

    f_sample_tran = fhn_transformation(f_params1)
    # f_sample_tran = lorenz_transformation(f_params)
    f_sample_dist = dirac_delta(f_sample_tran)

    g_sample_tran = linear_transformation(g_params)
    g_sample_dist = mvn(g_sample_tran, g_sample_cov)

    # Create train and test dataset
    hidden_train, obs_train, hidden_test, obs_test = \
        create_dataset(n_train, n_test, time_list, Dx, Dy, f_sample_dist, g_sample_dist, f_params_list, lb=-5, ub=5)
    print("finished creating dataset")

    hidden_all = np.concatenate([hidden_train, hidden_test])

    std = 0.1
    Dh = 5
    weights1 = np.random.randn(2, Dh)
    bias1 = np.random.randn(Dh)
    obs_train = np.maximum(np.matmul(hidden_train, weights1) + bias1, 0)
    obs_test = np.maximum(np.matmul(hidden_test, weights1) + bias1, 0)
    print(obs_train.shape, obs_test.shape)
    weights2 = np.random.randn(Dh, 1)
    bias2 = np.random.randn(1)
    obs_train = np.matmul(obs_train, weights2) + bias2 + np.random.randn(obs_train.shape[1], 1) * np.sqrt(std)
    obs_test = np.matmul(obs_test, weights2) + bias2 + np.random.randn(obs_test.shape[1], 1) * np.sqrt(std)
    print(obs_train.shape, obs_test.shape)

    datadict["Xtrue"] = hidden_all
    datadict["Ytrain"] = obs_train
    datadict["Yvalid"] = obs_test

    RLT_DIR = "../data/fhn/1D_NN_obs_{}_large_x0_range/".format(std)
    if not os.path.exists(RLT_DIR):
        os.makedirs(RLT_DIR)

    with open(RLT_DIR + "datadict", "wb") as f:
        pickle.dump(datadict, f)

    plot_fhn_results(RLT_DIR, hidden_all[0:10])
    plot_training_data(RLT_DIR, hidden_all[0:10], obs_train[0:10])

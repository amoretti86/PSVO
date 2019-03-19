import os

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np


def plot_fhn_results(RLT_DIR, Xs_val):
    if not os.path.exists(RLT_DIR + "/FHN 2D plots"):
        os.makedirs(RLT_DIR + "/FHN 2D plots")
    Xs_val = np.mean(Xs_val, axis=2)
    for i in range(Xs_val.shape[0]):
        plt.figure()
        plt.title("hidden state for all particles")
        plt.xlabel("x_dim 1")
        plt.ylabel("x_dim 2")
        plt.plot(Xs_val[i, :, 0], Xs_val[i, :, 1])
        sns.despine()
        plt.savefig(RLT_DIR + "/FHN 2D plots/All_x_paths_{}".format(i))
        plt.close()

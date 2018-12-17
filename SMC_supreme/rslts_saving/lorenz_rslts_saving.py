import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

import os


def plot_lorenz_results(RLT_DIR, Xs_val):
	if not os.path.exists(RLT_DIR + "/Lorenz 3D plots"):
		os.makedirs(RLT_DIR + "/Lorenz 3D plots")
    Xs_val = np.mean(Xs_val, axis=2)
	for i in range(Xs_val.shape[0]):        
		fig = plt.figure()
		ax = fig.gca(projection="3d")
		plt.title("hidden state for all particles")
		ax.set_xlabel("x_dim 1")
		ax.set_ylabel("x_dim 2")
		ax.set_zlabel("x_dim 3")
		ax.plot(Xs_val[i, :, 0], Xs_val[i, :, 1], Xs_val[i, :, 2])
		plt.savefig(RLT_DIR+"/Lorenz 3D plots/All_x_paths_{}".format(i))
		plt.close()

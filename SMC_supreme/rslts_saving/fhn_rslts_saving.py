import os

import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

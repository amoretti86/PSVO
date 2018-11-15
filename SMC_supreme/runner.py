import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf
import tensorflow_probability as tfp
import pdb

# for data saving stuff
import sys
import pickle
import json
import os

# import from files
from transformation.fhn import fhn_transformation, tf_fhn_transformation
from transformation.linear import linear_transformation, tf_linear_transformation
from transformation.lorenz import lorenz_transformation, tf_lorenz_transformation
from transformation.MLP import MLP_transformation

from distribution.dirac_delta import dirac_delta
from distribution.mvn import mvn, tf_mvn
from distribution.poisson import poisson, tf_poisson

from rslts_saving.rslts_saving import *

from encoder import encoder_cell
from sampler import create_dataset
from SMC import SMC

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # to avoid lots of log about the device

print("the code is written at:")
print("\ttensorflow version: 1.12.0")
print("\tensorflow_probability version: 0.5.0")

print("the system uses:")
print("\ttensorflow version:", tf.__version__)
print("\ttensorflow_probability version:", tfp.__version__)

if __name__ == "__main__":

	# ============================================ parameter part ============================================ #
	# training hyperparameters
	Dx = 2
	Dy = 3
	n_particles = 100
	time = 10

	batch_size = 16
	lr = 1e-3
	epoch = 10
	seed = 0
	tf.set_random_seed(seed)
	np.random.seed(seed)

	n_train = 5	* batch_size
	n_test  = 1 	* batch_size

	# printing and data saving params

	print_freq = 10

	store_res = True
	MSE_steps = 1
	save_freq = 10
	saving_num = min(n_train, 1*batch_size)
	rslt_dir_name = "some_name"
	
	# ============================================== model part ============================================== #
	# for data generation
	sigma, rho, beta, dt = 10.0, 28.0, 8.0/3.0, 0.01
	f_params = (sigma, rho, beta, dt)

	a, b, c, I, dt = 1.0, 0.95, 0.05, 1.0, 0.15
	f_params = (a, b, c, I, dt)

	f_cov = 0.15*np.eye(Dx)

	g_params = np.random.randn(Dy, Dx)
	g_cov = np.eye(Dy)

	# transformation can be: fhn_transformation, linear_transformation, lorenz_transformation
	# distribution can be: dirac_delta, mvn, poisson
	f_sample_tran = fhn_transformation(f_params)
	f_sample_dist = dirac_delta(f_sample_tran)

	g_sample_tran = linear_transformation(g_params)
	g_sample_dist = poisson(g_sample_tran)

	# for training
	x_0 = tf.placeholder(tf.float32, shape=(batch_size, Dx), name = "x_0")

	# q_train_tran = MLP_transformation([50], Dx, name="q_train_tran")
	# f_train_tran = MLP_transformation([50], Dx, name="f_train_tran")
	g_train_tran = MLP_transformation([50], Dy, name="g_train_tran")

	my_encoder_cell = encoder_cell(Dx, Dy, batch_size, time, name = "encoder_cell")
	q_train_tran = my_encoder_cell.q_transformation
	f_train_tran = my_encoder_cell.f_transformation

	q_train_dist = tf_mvn(q_train_tran, x_0, sigma_init=100, sigma_min=10, name="q_train_dist")
	f_train_dist = tf_mvn(f_train_tran, x_0, sigma_init=100, sigma_min=10, name="f_train_dist")
	g_train_dist = tf_poisson(g_train_tran, name="g_train_dist")

	# for evaluating log_ZSMC_true
	q_A = tf.eye(Dx)
	q_cov = 100*tf.eye(Dx)

	f_cov = 100*tf.eye(Dx)

	g_A = tf.constant(g_params, dtype = tf.float32)
	g_cov = tf.constant(g_cov, dtype = tf.float32)

	q_true_tran = tf_linear_transformation(q_A)
	q_true_dist = tf_mvn(q_true_tran, x_0, sigma=q_cov, name="q_true_dist")
	f_true_tran = tf_fhn_transformation(f_params)
	f_true_dist = tf_mvn(f_true_tran, x_0, sigma=f_cov, name="f_true_dist")
	g_true_tran = tf_linear_transformation(g_A)
	g_true_dist = tf_poisson(g_true_tran, name="g_true_dist")

	# =========================================== data saving part =========================================== #
	if store_res == True:
		Experiment_params = {"n_particles":n_particles, "time":time, "batch_size":batch_size,
							 "lr":lr, "epoch":epoch, "seed":seed, "n_train":n_train,
							 "rslt_dir_name":rslt_dir_name}
		print("Experiment_params")
		for key, val in Experiment_params.items():
			print("\t{}:{}".format(key, val))

		RLT_DIR = create_RLT_DIR(Experiment_params)
		print("RLT_DIR:", RLT_DIR)

		writer = tf.summary.FileWriter(RLT_DIR)

	log_ZSMC_trains = []
	log_ZSMC_tests = []
	MSE_trains = []
	MSE_tests = []

	# ============================================= dataset part ============================================= #
	# Create train and test dataset
	hidden_train, obs_train, hidden_test, obs_test = \
		create_dataset(n_train, n_test, time, Dx, Dy, f_sample_dist, g_sample_dist, lb=-3, ub=3)
	print("finish creating dataset")
	# if store_res == True:
	# 	plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num = saving_num)

	# ========================================== another model part ========================================== #

	# placeholders
	obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name = "obs")
	hidden = tf.placeholder(tf.float32, shape=(batch_size, time, Dx), name = "hidden")

	SMC_true  = SMC(q_true_dist,  f_true_dist,  g_true_dist,  n_particles, batch_size, name = "log_ZSMC_true")
	SMC_train = SMC(q_train_dist, f_train_dist, g_train_dist, n_particles, batch_size, encoder_cell = my_encoder_cell, name = "log_ZSMC_train")

	log_ZSMC_true,  log_true  = SMC_true.get_log_ZSMC(obs, x_0)
	log_ZSMC_train, log_train = SMC_train.get_log_ZSMC(obs, x_0)

	MSE_true, _, _ = SMC_true.n_step_MSE(MSE_steps, hidden, obs)
	MSE_train, ys_hat, _ = SMC_train.n_step_MSE(MSE_steps, hidden, obs)
	
	with tf.variable_scope("train"):
		train_op = tf.train.AdamOptimizer(lr).minimize(-log_ZSMC_train)	

	init = tf.global_variables_initializer()

	if store_res == True:
		saver = tf.train.Saver()
	with tf.Session() as sess:

		sess.run(init)

		if store_res == True:
			writer.add_graph(sess.graph)

		obs_all = np.concatenate((obs_train, obs_test))
		hidden_all = np.concatenate((hidden_train, hidden_test))
		log_ZSMC_true_val = SMC_true.tf_accuracy(sess, log_ZSMC_true, obs, obs_all, x_0, hidden_all)
		MSE_true_val = SMC_true.tf_MSE(sess, MSE_true, hidden, hidden_all, obs, obs_all)
		print("true log_ZSMC: {:<7.3f}, true MSE: {:>7.3f}".format(log_ZSMC_true_val, MSE_true_val))

		log_ZSMC_train_val = SMC_train.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
		log_ZSMC_test_val  = SMC_train.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
		MSE_train_val = SMC_train.tf_MSE(sess, MSE_train, hidden, hidden_train, obs, obs_train)
		MSE_test_val  = SMC_train.tf_MSE(sess, MSE_train, hidden, hidden_test,  obs, obs_test)
		print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}, train MSE: {:>7.3f}, test MSE: {:>7.3f}"\
			.format(0, log_ZSMC_train_val, log_ZSMC_test_val, MSE_train_val, MSE_test_val))


		log_ZSMC_trains.append(log_ZSMC_train_val)
		log_ZSMC_tests.append(log_ZSMC_test_val)
		MSE_trains.append(MSE_train_val)
		MSE_tests.append(MSE_test_val)

		for i in range(epoch):
			# train A, B, Q, x_0 using each training sample
			obs_train, hidden_train = shuffle(obs_train, hidden_train)
			for j in range(0, len(obs_train), batch_size):
				sess.run(train_op, feed_dict={obs:obs_train[j:j+batch_size], 
											  x_0:hidden_train[j:j+batch_size, 0]})

			# print training and testing loss
			if (i+1)%print_freq == 0:
				log_ZSMC_train_val = SMC_train.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
				log_ZSMC_test_val  = SMC_train.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
				MSE_train_val = SMC_train.tf_MSE(sess, MSE_train, hidden, hidden_train, obs, obs_train)
				MSE_test_val  = SMC_train.tf_MSE(sess, MSE_train, hidden, hidden_test,  obs, obs_test)
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}, train MSE: {:>7.3f}, test MSE: {:>7.3f}"\
					.format(i+1, log_ZSMC_train_val, log_ZSMC_test_val, MSE_train_val, MSE_test_val))

				log_ZSMC_trains.append(log_ZSMC_train_val)
				log_ZSMC_tests.append(log_ZSMC_test_val)
				MSE_trains.append(MSE_train_val)
				MSE_tests.append(MSE_test_val)

			if store_res == True and (i+1)%save_freq == 0:
				if not os.path.exists(RLT_DIR+"model/"): os.makedirs(RLT_DIR+"model/")
				saver.save(sess, RLT_DIR+"model/model_epoch", global_step=i+1)

		Xs = log_train[0]
		Xs_val = np.zeros((saving_num, time, n_particles, Dx))
		ys_hat_val = np.zeros((saving_num, MSE_steps, time - MSE_steps + 1, Dy))
		for i in range(0, saving_num, batch_size):
			X_val = sess.run(Xs, feed_dict = {obs:obs_train[i:i+batch_size],
											  x_0:hidden_train[j:j+batch_size, 0]})
			y_hat_val = sess.run(ys_hat, feed_dict = {obs:obs_train[i:i+batch_size],
													  hidden:hidden_train[i:i+batch_size]})
			for j in range(batch_size):
				Xs_val[i+j] = X_val[:, :, j, :]
				ys_hat_val[i+j] = y_hat_val[j]

	sess.close()

	print("finish training")

	if store_res == True:
		plot_training_data(RLT_DIR, hidden_train, obs_train, saving_num = saving_num)
		plot_learning_results(RLT_DIR, Xs_val, hidden_train, saving_num = saving_num)

		plot_fhn_results(RLT_DIR, Xs_val)
		plot_y_hat(RLT_DIR, ys_hat_val, obs_train)

		params_dict = {"time":time,
					   "n_particles":n_particles,
					   "batch_size":batch_size,
					   "lr":lr,
					   "epoch":epoch,
					   "n_train":n_train,
					   "seed":seed}
		true_model_dict = {"f_params":f_params,
						   "f_cov":f_cov,
						   "g_params":g_params,
						   "g_cov":g_cov}
		loss_dict = {"log_ZSMC_true":log_ZSMC_true_val,
					 "log_ZSMC_trains":log_ZSMC_trains,
					 "log_ZSMC_tests":log_ZSMC_tests,
					 "MSE_true":MSE_true_val,
					 "MSE_trains":MSE_trains,
					 "MSE_tests":MSE_tests}
		data_dict = {"params":params_dict,
					 "loss":loss_dict}

		with open(RLT_DIR + "data.json", "w") as f:
			json.dump(data_dict, f, indent = 4, cls = NumpyEncoder)
		train_data_dict = {"hidden_train":hidden_train[0:saving_num],
						   "obs_train":obs_train[0:saving_num]}
		learned_model_dict = {"Xs_val":Xs_val,
							  "ys_hat_val":ys_hat_val}
		data_dict["train_data_dict"] = train_data_dict
		data_dict["learned_model_dict"] = learned_model_dict

		with open(RLT_DIR + "data.p", "wb") as f:
			pickle.dump(data_dict, f)

		plot_log_ZSMC(RLT_DIR, log_ZSMC_true_val, log_ZSMC_trains, log_ZSMC_tests)
		plot_MSEs(RLT_DIR, MSE_true_val, MSE_trains, MSE_tests)
 
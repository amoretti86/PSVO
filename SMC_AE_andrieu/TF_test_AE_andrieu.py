import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf
import pdb

# import from files
from Encoder import Encoder
from andrieu_sampler import create_train_test_dataset
from distributions import mvn, poisson, tf_mvn, tf_poisson
from tf_andrieu import tf_andrieu_transition, tf_andrieu_emission
from SMC import SMC
from rslts_saving import create_RLT_DIR, NumpyEncoder, plot_training_data, plot_learning_results, plot_losses

# for data saving stuff
import sys
import pickle
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to reduce a lot of log about the device


print("Akwaaba!")
print(tf.__version__)

if __name__ == '__main__':

	# hyperparameters

	Dy, Dx = 1, 1
	n_particles = 10000
	time = 50

	batch_size = 5
	lr = 1e-4
	epoch = 50
	seed = 0

	n_train = 50	* batch_size
	n_test  = 1 	* batch_size

	print_freq = 10
	store_res = True
	rslt_dir_name = 'AutoEncoder_andrieu'

	tf.set_random_seed(seed)
	np.random.seed(seed)

	theta1 = 0.5
	theta2 = 0.05

	# create dir to store results
	if store_res == True:
		Experiment_params = {"n_particles":n_particles, "time":time, "batch_size":batch_size,
							 "lr":lr, "epoch":epoch, "seed":seed, "n_train":n_train,
							 "rslt_dir_name":rslt_dir_name}
		RLT_DIR = create_RLT_DIR(Experiment_params)

	# Create train and test dataset
	hidden_train, obs_train, hidden_test, obs_test = create_train_test_dataset(n_train, n_test, time, theta1, theta2)
	print("finish creating dataset")
	if store_res == True:
		plot_training_data(RLT_DIR, hidden_train, obs_train)
	# ================================ TF stuffs starts ================================ #

	# placeholders
	obs = tf.placeholder(tf.float32, shape=(batch_size, time), name = 'obs')
	obs_r = tf.expand_dims(obs, axis = -1, name = 'obs_r')

	encoder_cell = Encoder(Dx, n_particles, batch_size, time)

	# for evaluating true log_ZSMC
	q_true = tf_mvn(n_particles, batch_size, tf.eye(Dx), (10**2)*tf.eye(Dx), name = 'q_true')
	f_true = tf_andrieu_transition(n_particles, batch_size, theta1, name = 'f_true')
	g_true = tf_andrieu_emission(n_particles, batch_size, theta2, name = 'g_true')

	q_train = q_true
	f_train = f_true
	g_train = g_true

	# for train_op
	SMC_true  = SMC(q_true,  f_true,  g_true,  n_particles, batch_size, name = 'log_ZSMC_true')
	SMC_train = SMC(q_train, f_train, g_train, n_particles, batch_size, encoder_cell = encoder_cell, name = 'log_ZSMC_train')
	log_ZSMC_true,  log_true  = SMC_true.get_log_ZSMC(obs_r)
	log_ZSMC_train, log_train = SMC_train.get_log_ZSMC(obs_r)
	
	with tf.name_scope('train'):
		train_op = tf.train.AdamOptimizer(lr).minimize(-log_ZSMC_train)
	
	if store_res == True:
		writer = tf.summary.FileWriter(RLT_DIR)

	init = tf.global_variables_initializer()

	# for plot
	log_ZSMC_trains = []
	log_ZSMC_tests = []
	with tf.Session() as sess:

		sess.run(init)

		if store_res == True:
			writer.add_graph(sess.graph)

		log_ZSMC_true_val = SMC_true.tf_accuracy(sess, log_ZSMC_true, obs, obs_train+obs_test)
		print("log_ZSMC_true_val: {:<7.3f}".format(log_ZSMC_true_val))

		for i in range(epoch):
			# train A, B, Q, x_0 using each training sample
			obs_train, hidden_train = shuffle(obs_train, hidden_train)
			for j in range(0, len(obs_train), batch_size):
				sess.run(train_op, feed_dict={obs: obs_train[j:j+batch_size]})
				
			# print training and testing loss
			if (i+1)%print_freq == 0:
				log_ZSMC_train_val = SMC_true.tf_accuracy(sess, log_ZSMC_train, obs, obs_train)
				log_ZSMC_test_val  = SMC_true.tf_accuracy(sess, log_ZSMC_train, obs, obs_test)
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"\
					.format(i+1, log_ZSMC_train_val, log_ZSMC_test_val))

				log_ZSMC_trains.append(log_ZSMC_train_val)
				log_ZSMC_tests.append(log_ZSMC_test_val)

		Xs = log_train[0]
		Xs = log_true[0]
		As = log_train[-1]
		Xs_val = np.zeros((n_train, time, n_particles, Dx))
		As_val = np.zeros((n_train, time-1, Dx, Dx))
		for i in range(0, len(obs_train), batch_size):
			X_val, A_val = sess.run([Xs, As], feed_dict = {obs:obs_train[i:i+batch_size]})
			for j in range(batch_size):
				Xs_val[i+j] = X_val[:, :, j, :]
				As_val[i+j] = A_val[j]


	sess.close()
	print("finish training")

	if store_res == True:
		plot_training_data(RLT_DIR, hidden_train, obs_train)
		plot_learning_results(RLT_DIR, Xs_val, hidden_train)
		plot_losses(RLT_DIR, log_ZSMC_true_val, log_ZSMC_trains, log_ZSMC_tests)

		params_dict = {"T":time, "n_particles":n_particles, "batch_size":batch_size, "lr":lr,
				 	   "epoch":epoch, "n_train":n_train, "seed":seed}
		model_dict = {"theta1":theta1, "theta2":theta2}
		learned_model_dict = {"As_val":As_val}
		log_ZSMC_dict = {"log_ZSMC_true":log_ZSMC_true_val,
						 "log_ZSMC_trains": log_ZSMC_trains, 
						 "log_ZSMC_tests":log_ZSMC_tests}
		data_dict = {"params_dict":params_dict, 
					 "model_dict":model_dict, 
					 "learned_model_dict":learned_model_dict,
					 "log_ZSMC_dict":log_ZSMC_dict}
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)
		with open(RLT_DIR + 'data.json', 'w') as f:
			json.dump(data_dict, f, indent = 4, cls = NumpyEncoder)\

	print("fin")

 

		# Xs, log_Ws, Ws, fs, gs, qs, A_NbxTxDzxDz = log_true
		# obs_true = obs_train + obs_test
		# for i in range(0, len(obs_true), batch_size):
		# 	log_Ws_val, Ws_val, fs_val, gs_val, qs_val = sess.run([log_Ws, Ws, fs, gs, qs], 
		# 														  feed_dict={obs:obs_true[i:i+batch_size]})
		# 	for j, (log_W_nb, W_nb, f_nb, g_nb, q_nb) in enumerate(zip(log_Ws_val, Ws_val, fs_val, gs_val, qs_val)):
		# 		for k, (log_W_n, W_n, f_n, g_n, q_n) in enumerate(zip(log_W_nb.T, W_nb.T, f_nb.T, g_nb.T, q_nb.T)):
		# 			# pdb.set_trace()
		# 			print("trajectory:", i, "time:", j, "batch:", k)
		# 			idx = np.argsort(log_W_n)[-10:]
		# 			print("log_W_n")
		# 			print(log_W_n[idx])
		# 			print("W_n")
		# 			print(W_n[idx])
		# 			print("f_n")
		# 			print(f_n[idx])
		# 			print("g_n")
		# 			print(g_n[idx])
		# 			print("q_n")
		# 			print(q_n[idx])
		# 			print("np.log(np.mean(W_n))")
		# 			print(np.log(np.mean(W_n)))
		# 			print()
		# 			assert math.isfinite(np.log(np.mean(W_n)))
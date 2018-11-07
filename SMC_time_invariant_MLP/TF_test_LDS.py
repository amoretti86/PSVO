import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf
import pdb

# import from files
from MLP import MLP_mvn, MLP_poisson
from SMC_sampler import create_train_test_dataset
from distributions import mvn, poisson, tf_mvn, tf_poisson
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

	Dy, Dx = 2, 2
	n_particles = 500
	time = 100

	batch_size = 5
	lr = 5e-3
	epoch = 100
	seed = 0

	n_train = 100	* batch_size
	n_test  = 1 	* batch_size

	print_freq = 10
	store_res = True
	save_freq = 10
	max_fig_num = 20
	rslt_dir_name = 'time_invariant_MLP_LDS'

	tf.set_random_seed(seed)
	np.random.seed(seed)
	
	A_true = np.diag([0.95, 0.25])
	Q_true = np.asarray([[1., 0], [0, 1.]])
	B_true = np.diag([2.0, 4.0])
	Sigma_true = np.asarray([[1., 0], [0, 1.]])
	x_0_true = np.array([1.0, 1.0])

	A_init = A_true 				# np.diag([0.5, 0.95])
	B_init = B_true 				# np.diag([1.5, 1.5])
	L_Q_init = Q_true 				# np.asarray([[1.2, 0], [0, 1.2]]) # Q = L * L^T
	L_Sigma_init = Sigma_true 		# np.asarray([[1.2, 0], [0, 1.2]]) # Q = L * L^T
	x_0_init = x_0_true 			# np.array([0.8, 0.8])

	# create dir to store results
	if store_res == True:
		Experiment_params = {"n_particles":n_particles, "time":time, "batch_size":batch_size,
							 "lr":lr, "epoch":epoch, "seed":seed, "n_train":n_train,
							 "rslt_dir_name":rslt_dir_name}
		print('Experiment_params')
		for key, val in Experiment_params.items():
			print('\t{}:{}'.format(key, val))
		RLT_DIR = create_RLT_DIR(Experiment_params)
		print("RLT_DIR:", RLT_DIR)

	# Create train and test dataset
	f = mvn(A_true, Q_true, x_0_true)
	g = mvn(B_true, Sigma_true)
	hidden_train, obs_train, hidden_test, obs_test = create_train_test_dataset(n_train, n_test, time, x_0_true, f, g, Dx, Dy)
	print("finish creating dataset")
	# ================================ TF stuffs starts ================================ #

	# placeholders
	obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name = 'obs')
	x_0 = tf.placeholder(tf.float32, shape=(batch_size, Dx), name = 'x_0')

	# for evaluating true log_ZSMC
	A_true_tnsr 	= tf.Variable(A_true, 		dtype=tf.float32, trainable = False, name='A_true')
	B_true_tnsr 	= tf.Variable(B_true, 		dtype=tf.float32, trainable = False, name='B_true')
	Q_true_tnsr 	= tf.Variable(Q_true, 		dtype=tf.float32, trainable = False, name='Q_true')
	Sigma_true_tnsr = tf.Variable(Sigma_true, 	dtype=tf.float32, trainable = False, name='Q_true')
	x_0_true_tnsr 	= tf.Variable(x_0_true, 	dtype=tf.float32, trainable = False, name='x_0_true')
	q_true = tf_mvn(n_particles, batch_size, tf.eye(Dx),  (5)*tf.eye(Dx), x_0, 	name = 'q_true')
	f_true = tf_mvn(n_particles, batch_size, A_true_tnsr, Q_true_tnsr, x_0, 	name = 'f_true')
	g_true = tf_mvn(n_particles, batch_size, B_true_tnsr, Sigma_true_tnsr, 		name = 'g_true')

	# A, B, Q, x_0 to train
	# A 		= tf.Variable(A_init, 		dtype=tf.float32, trainable = False, name='A')
	# B 		= tf.Variable(B_init, 		dtype=tf.float32, trainable = False, name='B')
	# L_Q 	= tf.Variable(L_Q_init, 	dtype=tf.float32, trainable = False, name='L_Q')
	# L_Sigma = tf.Variable(L_Sigma_init, dtype=tf.float32, trainable = False, name='L_Sigma')
	# x_0 	= tf.Variable(x_0_init, 	dtype=tf.float32, trainable = False, name='x_0')
	# Q 		= tf.matmul(L_Q, 	 L_Q, 	  transpose_b = True, name = 'Q')
	# Sigma 	= tf.matmul(L_Sigma, L_Sigma, transpose_b = True, name = 'Sigma')

	q_train = MLP_mvn(Dx + Dy, Dx, n_particles, batch_size, name = 'q_train')
	f_train = MLP_mvn(Dx, Dx, n_particles, batch_size, name = 'f_train')
	g_train = MLP_mvn(Dx, Dy, n_particles, batch_size, name = 'g_train')

	# for train_op
	SMC_true  = SMC(q_true,  f_true,  g_true,  n_particles, batch_size, name = 'log_ZSMC_true')
	SMC_train = SMC(q_train, f_train, g_train, n_particles, batch_size, name = 'log_ZSMC_train')
	log_ZSMC_true,  log_true  = SMC_true.get_log_ZSMC(obs, x_0)
	log_ZSMC_train, log_train = SMC_train.get_log_ZSMC(obs, x_0)
	
	with tf.name_scope('train'):
		train_op = tf.train.AdamOptimizer(lr).minimize(-log_ZSMC_train)
	
	if store_res == True:
		writer = tf.summary.FileWriter(RLT_DIR)
		saver = tf.train.Saver()

	# check trainable variables
	# print('trainable variables:')
	# for var in tf.trainable_variables():
	# 	print(var)

	init = tf.global_variables_initializer()

	# for plot
	log_ZSMC_trains = []
	log_ZSMC_tests = []
	with tf.Session() as sess:

		sess.run(init)

		if store_res == True:
			writer.add_graph(sess.graph)

		log_ZSMC_true_val = SMC_true.tf_accuracy(sess, log_ZSMC_true, obs, obs_train+obs_test, x_0, hidden_train+hidden_test)
		print("log_ZSMC_true_val: {:<7.3f}".format(log_ZSMC_true_val))
		log_ZSMC_train_val = SMC_true.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
		log_ZSMC_test_val  = SMC_true.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
		print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"\
			.format(0, log_ZSMC_train_val, log_ZSMC_test_val))
		log_ZSMC_trains.append(log_ZSMC_train_val)
		log_ZSMC_tests.append(log_ZSMC_test_val)

		for i in range(epoch):
			# train A, B, Q, x_0 using each training sample
			obs_train, hidden_train = shuffle(obs_train, hidden_train)
			for j in range(0, len(obs_train), batch_size):
				sess.run(train_op, feed_dict={obs:obs_train[j:j+batch_size], 
											  x_0:[hidden[0] for hidden in hidden_train[j:j+batch_size]]})
				
			# print training and testing loss
			if (i+1)%print_freq == 0:
				log_ZSMC_train_val = SMC_true.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
				log_ZSMC_test_val  = SMC_true.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"\
					.format(i+1, log_ZSMC_train_val, log_ZSMC_test_val))

				log_ZSMC_trains.append(log_ZSMC_train_val)
				log_ZSMC_tests.append(log_ZSMC_test_val)

			if store_res == True and (i+1)%save_freq == 0:
				saver.save(sess, os.path.join(RLT_DIR, 'model/model_epoch'), global_step=i+1)

		Xs = log_train[0]
		Xs_val = np.zeros((n_train, time, n_particles, Dx))
		for i in range(0, min(len(hidden_train), max_fig_num), batch_size):
			X_val = sess.run(Xs, feed_dict = {obs:obs_train[i:i+batch_size],
											  x_0:[hidden[0] for hidden in hidden_train[i:i+batch_size]]})
			for j in range(batch_size):
				Xs_val[i+j] = X_val[:, :, j, :]

	sess.close()

	print("fin")

	if store_res == True:
		plot_training_data(RLT_DIR, hidden_train, obs_train, max_fig_num = max_fig_num)
		plot_learning_results(RLT_DIR, Xs_val, hidden_train, max_fig_num = max_fig_num)
		plot_losses(RLT_DIR, log_ZSMC_true_val, log_ZSMC_trains, log_ZSMC_tests)

		params_dict = {"T":time, "n_particles":n_particles, "batch_size":batch_size, "lr":lr,
					   "epoch":epoch, "n_train":n_train, "seed":seed}
		true_model_dict = { "A_true":A_true, "Q_true":Q_true, 
							"B_true":B_true, "x_0_true":x_0_true}
		learned_model_dict = {"Xs_val":Xs_val}
		log_ZSMC_dict = {"log_ZSMC_true":log_ZSMC_true_val, 
						 "log_ZSMC_trains": log_ZSMC_trains, 
						 "log_ZSMC_tests":log_ZSMC_tests}
		data_dict = {"params_dict":params_dict, 
					 "true_model_dict":true_model_dict, 
					 "learned_model_dict":learned_model_dict,
					 "log_ZSMC_dict":log_ZSMC_dict}
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)
		with open(RLT_DIR + 'data.json', 'w') as f:
			json.dump(data_dict, f, indent = 4, cls = NumpyEncoder)
 
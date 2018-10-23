import numpy as np
import scipy as sp
import random
import math

import tensorflow as tf
import pdb

# import from files
from SMC_sampler import *
from distributions import *
from posterior_approx import *
from SMC import *
from rslts_saving import *

# for data saving stuff
import os
import pickle
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to reduce a lot of log about the device


print("Akwaaba!")
print(tf.__version__)

def evaluate_mean_log_ZSMC(log_ZSMC, obs_samples):
	"""
	used for evaluating true_log_ZSMC, train_log_ZSMC, test_log_ZSMC
	"""
	log_ZSMCs = []
	for obs_sample in obs_samples:
		log_ZSMC_val = sess.run(log_ZSMC, feed_dict={obs: obs_sample})
		log_ZSMCs.append(log_ZSMC_val)

	return np.mean(log_ZSMCs)

if __name__ == '__main__':

	# hyperparameters
	n_particles = 10000
	n_iters = 2         # num_iters for Laplace Approx
	time = 50
	lr = 1e-4
	epoch = 50
	seed = 0

	n_train = 100
	n_test = 5

	use_stop_gradient = False
	print_freq = 10
	store_res = True
	rslt_dir_name = 'only_learn_A'
	rslt_dir_name = 'only_learn_A_poisson'

	tf.set_random_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	
	A_true = np.asarray([[0.85, -0.25], [-0.05, 0.54]])
	Q_true = np.asarray([[1., 0], [0, 1.]])
	B_true = np.diag([1.0, 2.0])
	Sigma_true = np.asarray([[1., 0], [0, 1.]])
	x_0_true = np.array([5.0, 5.0])

	Dy, Dx = B_true.shape

	# whether create dir and store results
	if store_res == True:
		Experiment_params = {"n_particles":n_particles, "n_iters":n_iters, "time":time, 
							 "lr":lr, "epoch":epoch, "seed":seed, "n_train":n_train, 
							 "use_stop_gradient":use_stop_gradient,
							 "rslt_dir_name":rslt_dir_name}
		RLT_DIR = create_RLT_DIR(Experiment_params)


	# Create train and test dataset
	f = multivariate_normal(A_true, Q_true, x_0_true)
	# g = multivariate_normal(B_true, Sigma_true)
	g = poisson(B_true)
	hidden_train, obs_train, hidden_test, obs_test = create_train_test_dataset(n_train, n_test, time, x_0_true, f, g, Dx, Dy)
	print("finish creating dataset")

	# init A, B, Q, Sigma, x_0 randomly
	A_init = np.diag([0.5, 0.95]) 	# A_true
	B_init = B_true 				# np.diag([10.0, 10.0])
	L_Q_init = Q_true 				# np.asarray([[1.2, 0], [0, 1.2]]) # Q = L * L^T
	L_Sigma_init = Sigma_true 		# np.asarray([[1.2, 0], [0, 1.2]]) # Q = L * L^T
	x_0_init = x_0_true 			# np.array([0.8, 0.8])

	# ======================================= TF stuffs starts ======================================= #
	obs = tf.placeholder(tf.float32, shape=(time, Dy), name = 'obs')

	# for evaluating true log_ZSMC
	A_true_tnsr 	= tf.Variable(A_true, 		dtype=tf.float32, trainable = False, name='A_true')
	B_true_tnsr 	= tf.Variable(B_true, 		dtype=tf.float32, trainable = False, name='B_true')
	Q_true_tnsr 	= tf.Variable(Q_true, 		dtype=tf.float32, trainable = False, name='Q_true')
	Sigma_true_tnsr = tf.Variable(Sigma_true, 	dtype=tf.float32, trainable = False, name='Q_true')
	x_0_true_tnsr 	= tf.Variable(x_0_true, 	dtype=tf.float32, trainable = False, name='x_0_true')

	q_true = tf_multivariate_normal(n_particles, tf.eye(Dx),  (20**2)*tf.eye(Dx), 			name = 'q_true')
	f_true = tf_multivariate_normal(n_particles, A_true_tnsr, Q_true_tnsr, x_0_true_tnsr, 	name = 'f_true')
	# g_true = tf_multivariate_normal(n_particles, B_true_tnsr, Sigma_true_tnsr, 			  	name = 'g_true')
	g_true = tf_poisson(n_particles, B_true_tnsr, name = 'g_true')
	p_true = TensorGaussianPostApprox(A_true_tnsr, B_true_tnsr, Q_true_tnsr, Sigma_true_tnsr, name = 'p_true')

	# A, B, Q, x_0 to train
	A 		= tf.Variable(A_init, 		dtype=tf.float32, name='A')
	B 		= tf.Variable(B_init, 		dtype=tf.float32, trainable = False, name='B')
	L_Q 	= tf.Variable(L_Q_init, 	dtype=tf.float32, trainable = False, name='L_Q')
	L_Sigma = tf.Variable(L_Sigma_init, dtype=tf.float32, trainable = False, name='L_Sigma')
	x_0 	= tf.Variable(x_0_init, 	dtype=tf.float32, trainable = False, name='x_0')
	Q 		= tf.matmul(L_Q, 	 L_Q, 	  transpose_b = True, name = 'Q')
	Sigma 	= tf.matmul(L_Sigma, L_Sigma, transpose_b = True, name = 'Sigma')

	q_train = tf_multivariate_normal(n_particles, tf.eye(Dx), (20**2)*tf.eye(Dx), name = 'q_train')
	f_train = tf_multivariate_normal(n_particles, A, 		  Q, x_0, 	  name = 'f_train')
	# g_train = tf_multivariate_normal(n_particles, B, 		  Sigma, 	  name = 'g_train')
	g_train = tf_poisson(n_particles, B, name = 'g_true')
	p_train = TensorGaussianPostApprox(A, B, Q, Sigma, name = 'p_train')

	# true_log_ZSMC: log_ZSMC generated from true A, B, Q, x_0
	log_ZSMC_true, log_true = get_log_ZSMC(obs, n_particles, q_true,  f_true,  g_true,  p_true,  
										   use_stop_gradient, name = 'log_ZSMC_true')
	log_ZSMC, log		 	= get_log_ZSMC(obs, n_particles, q_train, f_train, g_train, p_train, 
										   use_stop_gradient, name = 'log_ZSMC_train')
	with tf.name_scope('train'):
		train_op = tf.train.GradientDescentOptimizer(lr).minimize(-log_ZSMC)

	# tf summary writer
	if store_res == True:
		writer = tf.summary.FileWriter(RLT_DIR)

	init = tf.global_variables_initializer()

	log_ZSMC_trains = []
	log_ZSMC_tests = []
	with tf.Session() as sess:

		sess.run(init)

		writer.add_graph(sess.graph)

		true_log_ZSMC_val = evaluate_mean_log_ZSMC(log_ZSMC_true, obs_train + obs_test)
		print("true_log_ZSMC_val: {:<10.4f}".format(true_log_ZSMC_val))

		for i in range(epoch):
			# train A, B, Q, x_0 using each training sample
			# np.random.shuffle(obs_train)
			for j, obs_sample in enumerate(obs_train):
				sess.run(train_op, feed_dict={obs: obs_sample})

			# print training and testing loss
			if (i+1)%print_freq == 0:
				log_ZSMC_train = evaluate_mean_log_ZSMC(log_ZSMC, obs_train)
				log_ZSMC_test  = evaluate_mean_log_ZSMC(log_ZSMC, obs_test)
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"\
					.format(i+1, log_ZSMC_train, log_ZSMC_test))
				log_ZSMC_trains.append(log_ZSMC_train)
				log_ZSMC_tests.append(log_ZSMC_test)

				print("A")
				print(A.eval())

		A_val = A.eval()
		B_val = B.eval()
		Q_val = Q.eval()
		Sigma_val = Sigma.eval()
		x_0_val = x_0.eval()

		Xs = log[0]
		Xs_val = np.zeros((n_train, time, n_particles, Dx))
		for i in range(n_train):
			X_val = sess.run(Xs, feed_dict = {obs:obs_train[i]})
			Xs_val[i] = X_val

	sess.close()

	if store_res == True:
		plot_training_data(RLT_DIR, hidden_train, obs_train)
		plot_learning_results(RLT_DIR, Xs_val, hidden_train)
		plot_losses(RLT_DIR, true_log_ZSMC_val, log_ZSMC_trains, log_ZSMC_tests)

		params_dict = {"n_particles":n_particles, "n_iters":n_iters, "time":time, "lr":lr, "epoch":epoch, \
					   "seed":seed, "n_train":n_train, "use_stop_gradient":use_stop_gradient}
		true_model_dict = { "A_true":A_true, "Q_true":Q_true, 
							"B_true":B_true, "x_0_true":x_0_true}
		init_model_dict = { "A_init":A_init, "Q_init":np.dot(L_Q_init, L_Q_init.T), 
							"B_init":B_init, "x_0_init":x_0_init}
		learned_model_dict = {"A_val":A_val, "Q_val":Q_val, 
							  "B_val":B_val, "x_0_val":x_0_val}
		log_ZSMC_dict = {"true_log_ZSMC":true_log_ZSMC_val, "log_ZSMC_trains": log_ZSMC_trains, 
						 "log_ZSMC_tests":log_ZSMC_tests}
		data_dict = {"params_dict":params_dict, 
					 "true_model_dict":true_model_dict,
					 "init_model_dict":init_model_dict,
					 "learned_model_dict":learned_model_dict, 
					 "log_ZSMC_dict":log_ZSMC_dict}
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)
		with open(RLT_DIR + 'data.json', 'w') as f:
			json.dump(data_dict, f, indent = 4, cls = NumpyEncoder)

	print("fin")

	print("-------------------true val-------------------")
	print("A_true")
	print(A_true)
	print("Q_true")
	print(Q_true)
	print("B_true")
	print(B_true)
	print("Sigma_true")
	print(Sigma_true)
	print("x_0_true")
	print(x_0_true)
	print("-------------------optimized val-------------------")
	print("A_val")
	print(A_val)
	print("Q_val")
	print(Q_val)
	print("B_val")
	print(B_val)
	print("Sigma_val")
	print(Sigma_val)
	print("x_0_val")
	print(x_0_val)

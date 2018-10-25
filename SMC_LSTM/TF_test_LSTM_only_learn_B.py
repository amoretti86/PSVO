import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf
import pdb

# import from files
from VRNN import VartiationalRNN
from SMC_sampler import create_train_test_dataset
from distributions import mvn, poisson, tf_mvn, tf_poisson
from posterior_approx import *
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
	Dh = 50
	n_particles = 1000
	time = 50

	batch_size = 5
	lr = 1e-4
	epoch = 100
	seed = 0

	n_train = 100	* batch_size
	n_test  = 1 	* batch_size

	print_freq = 10
	store_res = True
	rslt_dir_name = 'only_learn_B'

	tf.set_random_seed(seed)
	np.random.seed(seed)
	
	A_true = np.diag([0.95, 0.25])
	Q_true = np.asarray([[1., 0], [0, 1.]])
	B_true = np.diag([1.0, 2.0])
	Sigma_true = np.asarray([[1., 0], [0, 1.]])
	x_0_true = np.array([1.0, 1.0])

	# create dir to store results
	if store_res == True:
		Experiment_params = (Dh, time, n_particles, batch_size, lr, epoch, seed, n_train)
		Experiment_params = {"n_particles":n_particles, "time":time, "Dh":Dh,
							 "lr":lr, "epoch":epoch, "seed":seed, "n_train":n_train,
							 "rslt_dir_name":rslt_dir_name}
		RLT_DIR = create_RLT_DIR(Experiment_params)

	# Create train and test dataset
	f = mvn(A_true, Q_true, x_0_true)
	g = mvn(B_true, Sigma_true)
	hidden_train, obs_train, hidden_test, obs_test = create_train_test_dataset(n_train, n_test, time, x_0_true, f, g, Dx, Dy)
	print("finish creating dataset")

	A_init = A_true 				# np.diag([0.5, 0.95])
	B_init = B_true 				# np.diag([1.5, 1.5])
	L_Q_init = Q_true 				# np.asarray([[1.2, 0], [0, 1.2]]) # Q = L * L^T
	L_Sigma_init = Sigma_true 		# np.asarray([[1.2, 0], [0, 1.2]]) # Q = L * L^T
	x_0_init = x_0_true 			# np.array([0.8, 0.8])

	# ================================ TF stuffs starts ================================ #

	# placeholders
	obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name = 'obs')
	obs_set_true  = tf.placeholder(tf.float32, shape=(n_train + n_test, time, Dy), 	name = 'obs_set_true')
	obs_set_train = tf.placeholder(tf.float32, shape=(n_train, 			time, Dy), 	name = 'obs_set_train')
	obs_set_test  = tf.placeholder(tf.float32, shape=(n_test, 			time, Dy), 	name = 'obs_set_test')

	VRNN_Cell = VartiationalRNN(Dx, Dy, Dh, n_particles, batch_size)

	# for evaluating true log_ZSMC
	A_true_tnsr 	= tf.Variable(A_true, 		dtype=tf.float32, trainable = False, name='A_true')
	B_true_tnsr 	= tf.Variable(B_true, 		dtype=tf.float32, trainable = False, name='B_true')
	Q_true_tnsr 	= tf.Variable(Q_true, 		dtype=tf.float32, trainable = False, name='Q_true')
	Sigma_true_tnsr = tf.Variable(Sigma_true, 	dtype=tf.float32, trainable = False, name='Q_true')
	x_0_true_tnsr 	= tf.Variable(x_0_true, 	dtype=tf.float32, trainable = False, name='x_0_true')
	q_true = tf_mvn(n_particles, batch_size, tf.eye(Dx),  (10**2)*tf.eye(Dx), 		  name = 'q_true')
	f_true = tf_mvn(n_particles, batch_size, A_true_tnsr, Q_true_tnsr, x_0_true_tnsr, name = 'f_true')
	g_true = tf_mvn(n_particles, batch_size, B_true_tnsr, Sigma_true_tnsr, 			  name = 'g_true')
	p_true = TensorGaussianPostApprox(A_true_tnsr, B_true_tnsr, Q_true_tnsr, Sigma_true_tnsr, name = 'p_true')

	# A, B, Q, x_0 to train
	A 		= tf.Variable(A_init, 		dtype=tf.float32, trainable = False, name='A', )
	B 		= tf.Variable(B_init, 		dtype=tf.float32, trainable = False, name='B')
	L_Q 	= tf.Variable(L_Q_init, 	dtype=tf.float32, trainable = False, name='L_Q')
	L_Sigma = tf.Variable(L_Sigma_init, dtype=tf.float32, trainable = False, name='L_Sigma')
	x_0 	= tf.Variable(x_0_init, 	dtype=tf.float32, trainable = False, name='x_0')
	Q 		= tf.matmul(L_Q, 	 L_Q, 	  transpose_b = True, name = 'Q')
	Sigma 	= tf.matmul(L_Sigma, L_Sigma, transpose_b = True, name = 'Sigma')

	q_train = tf_mvn(n_particles, batch_size, tf.eye(Dx), (10**2)*tf.eye(Dx), 	name = 'q_train')
	f_train = tf_mvn(n_particles, batch_size, A, 		  Q, x_0, 	  			name = 'f_train')
	g_train = None # tf_mvn(n_particles, batch_size, B, 		  Sigma, 	  			name = 'g_train')
	p_train = TensorGaussianPostApprox(A, B, Q, Sigma, name = 'p_train') # Revisit me

	# for train_op
	SMC_true  = SMC(VRNN_Cell, q_true,  f_true,  g_true,  p_true,  name = 'log_ZSMC_true')
	SMC_train = SMC(VRNN_Cell, q_train, f_train, g_train, p_train, name = 'log_ZSMC_train')
	log_ZSMC_true,  log_true  = SMC_true.get_log_ZSMC(obs)
	log_ZSMC_train, log_train = SMC_train.get_log_ZSMC(obs)
	
	with tf.name_scope('train'):
		train_op = tf.train.GradientDescentOptimizer(lr).minimize(-log_ZSMC_train)
	
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

		log_ZSMC_true_val = SMC_true.tf_accuracy(obs_train + obs_test, obs, log_ZSMC_true, sess, batch_size)
		print("log_ZSMC_true_val: {:<7.3f}".format(log_ZSMC_true_val))

		for i in range(epoch):
			# train A, B, Q, x_0 using each training sample
			obs_train, hidden_train = shuffle(obs_train, hidden_train)
			for j in range(0, len(obs_train), batch_size):
				sess.run(train_op, feed_dict={obs: obs_train[j:j+batch_size]})
				
			# print training and testing loss
			if (i+1)%print_freq == 0:
				log_ZSMC_train_val = SMC_true.tf_accuracy(obs_train, obs, log_ZSMC_train, sess, batch_size)
				log_ZSMC_test_val  = SMC_true.tf_accuracy(obs_test,  obs, log_ZSMC_train, sess, batch_size)
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"\
					.format(i+1, log_ZSMC_train_val, log_ZSMC_test_val))

				log_ZSMC_trains.append(log_ZSMC_train_val)
				log_ZSMC_tests.append(log_ZSMC_test_val)

		A_val = A.eval()
		B_val = B.eval()
		Q_val = Q.eval()
		x_0_val = x_0.eval()

		Xs = log_train[0]
		Xs_val = np.zeros((n_train, time, n_particles, Dx))
		for i in range(0, len(obs_train), batch_size):
			X_val = sess.run(Xs, feed_dict = {obs:obs_train[i:i+batch_size]})
			for j in range(batch_size):
				Xs_val[i+j] = X_val[:, :, j, :]

	sess.close()

	print("fin")

	print("-------------------true val-------------------")
	print("A_true")
	print(A_true)
	print("Q_true")
	print(Q_true)
	print("B_true")
	print(B_true)
	print("x_0_true")
	print(x_0_true)
	print("-------------------optimized val-------------------")
	print("A_val")
	print(A_val)
	print("Q_val")
	print(Q_val)
	print("B_val")
	print(B_val)
	print("x_0_val")
	print(x_0_val)

	if store_res == True:
		plot_training_data(RLT_DIR, hidden_train, obs_train)
		plot_learning_results(RLT_DIR, Xs_val, hidden_train)
		plot_losses(RLT_DIR, log_ZSMC_true_val, log_ZSMC_trains, log_ZSMC_tests)

		params_dict = {"Dh": Dh, "T":time, "n_particles":n_particles, "batch_size":batch_size, "lr":lr,
				 	   "epoch":epoch, "n_train":n_train, "seed":seed}
		true_model_dict = { "A_true":A_true, "Q_true":Q_true, 
							"B_true":B_true, "x_0_true":x_0_true}
		init_model_dict = { "A_init":A_init, "Q_init":np.dot(L_Q_init, L_Q_init.T), 
							"B_init":B_init, "x_0_init":x_0_init}
		learned_model_dict = {"A_val":A_val, "Q_val":Q_val, 
							  "B_val":B_val, "x_0_val":x_0_val}
		log_ZSMC_dict = {"log_ZSMC_true":log_ZSMC_true_val, "log_ZSMC_trains": log_ZSMC_trains, 
						 "log_ZSMC_tests":log_ZSMC_tests}
		data_dict = {"params_dict":params_dict, "true_model_dict":true_model_dict, 
					 "learned_model_dict":learned_model_dict, "log_ZSMC_dict":log_ZSMC_dict}
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)
		with open(RLT_DIR + 'data.json', 'w') as f:
			json.dump(data_dict, f, indent = 4, cls = NumpyEncoder)
 
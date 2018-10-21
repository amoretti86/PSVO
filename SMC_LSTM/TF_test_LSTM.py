import numpy as np
import scipy as sp
import random
import math

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import pdb

# import from files
from SMC_sampler import SMC_sampler
from distributions import multivariate_normal, poisson, tf_multivariate_normal, tf_poisson
from posterior_approx import LaplaceApprox, GaussianPostApprox, TensorLaplaceApprox, TensorGaussianPostApprox
from VartiationalRNN import VartiationalRNN

# for data saving stuff
import sys
import pickle
import os
from datetime import datetime
from datetools import addDateTime
from optparse import OptionParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to reduce a lot of log about the device


print("Akwaaba!")
print(tf.__version__)

def get_log_ZSMC(obs, VRNN_Cell, q, f, g, p, is_train = False, name = "get_log_ZSMC", debug_mode = False):
	"""
	Input:
		obs.shape = (batch_size, time, Dy)
		VRNN_Cell: an instance of VartiationalRNN
		is_train: False when computing true_log_ZSMC
				  True when computing train_log_ZSMC
	Output:
		log_ZSMC: shape = (batch_size,)
		log: stuff to debug
	"""
	with tf.name_scope(name):
		Dx = VRNN_Cell.Dx
		n_particles = VRNN_Cell.n_particles
		batch_size, time, Dy = obs.get_shape().as_list()

		if debug_mode:
			# please manually adjust values here
			assert Dx 			== 2, 	"correct Dx: {}, wrong Dx: {}".format(2, Dx)
			assert n_particles 	== 500, "correct n_particles: {}, wrong n_particles: {}".format(500, n_particles)
			assert batch_size 	== 5, 	"correct batch_size: {}, wrong batch_size: {}".format(5, batch_size)
			assert time 		== 25, 	"correct time: {}, wrong time: {}".format(25, time)
			assert Dy 			== 2, 	"correct Dy: {}, wrong Dy: {}".format(2, Dy)

		Xs = []
		Ws = []
		W_means = []
		fs = []
		gs = []
		qs = []
		ps = []

		# time = 1
		VRNN_Cell.reset()

		if is_train:

			VRNN_Cell.step(obs[:,0])
			X = VRNN_Cell.get_q_sample() 					# X.shape = (n_particles, batch_size, Dx)
			q_uno_probs = VRNN_Cell.get_q_prob(X)			# probs.shape = (n_particles, batch_size)
			f_nu_probs  = VRNN_Cell.get_f_prob(X)
			g_uno_probs = VRNN_Cell.get_g_prob(X, obs[:,0])

		else:
			X = q.sample(None, name = 'X0')
			q_uno_probs = q.prob(None, X, name = 'q_uno_probs')
			f_nu_probs  = f.prob(None, X, name = 'f_nu_probs')
			g_uno_probs = g.prob(X, obs[:,0], name = 'g_uno_probs')

		W = tf.multiply(g_uno_probs, f_nu_probs / q_uno_probs, name = 'W_0') 	# (n_particles, batch_size)
		log_ZSMC = tf.log(tf.reduce_mean(W, axis = 0, name = 'W_0_mean'), 		# (batch_size,)
						  name = 'log_ZSMC_0')

		Xs.append(X)
		Ws.append(W)
		W_means.append(tf.reduce_mean(W, axis = 0))
		fs.append(f_nu_probs)
		gs.append(g_uno_probs)
		qs.append(q_uno_probs)
		ps.append(tf.zeros(n_particles, batch_size))

		for t in range(1, time):

			# W_{t-1} = W_{t-1} * p(y_t | X_{t-1})
			# k = p.posterior(X, obs[t], name = 'p_{}'.format(t))
			k = tf.ones((n_particles, batch_size), dtype = tf.float32, name = 'p_{}'.format(t))
			W = W * k
			W = W/tf.reduce_sum(W, axis = 0)
			W = tf.transpose(W)																		# (batch_size, n_particles)
			
			categorical = tfd.Categorical(probs = W, name = 'Categorical_{}'.format(t))
			# idx = tf.stop_gradient(categorical.sample(n_particles))								# (n_particles, batch_size)
			idx = categorical.sample(n_particles)

			# ugly stuff used to resample X
			ugly_stuff = tf.tile(tf.expand_dims(tf.range(batch_size), axis = 0), (n_particles, 1)) 	# (n_particles, batch_size)
			idx_expanded = tf.expand_dims(idx, axis = 2)											# (n_particles, batch_size, 1)
			ugly_expanded = tf.expand_dims(ugly_stuff, axis = 2)									# (n_particles, batch_size, 1)
			final_idx = tf.concat((idx_expanded, ugly_expanded), axis = 2)							# (n_particles, batch_size, 2)
			X_prev = tf.gather_nd(X, final_idx)														# (n_particles, batch_size, Dx)
					
			VRNN_Cell.update_lstm(X_prev, obs[:,t-1])

			if is_train:
				VRNN_Cell.step(obs[:,t])
				X = VRNN_Cell.get_q_sample() 					# X.shape = (n_particles, batch_size, Dx)
				q_t_probs = VRNN_Cell.get_q_prob(X)				# probs.shape = (n_particles, batch_size)				
				f_t_probs = VRNN_Cell.get_f_prob(X)
				g_t_probs = VRNN_Cell.get_g_prob(X, obs[:,t])
			else:
				X = q.sample(X_prev, name = 'q_{}_sample'.format(t))
				q_t_probs = q.prob(X_prev, X, name = 'q_{}_probs'.format(t))
				f_t_probs = f.prob(X_prev, X, name = 'f_{}_probs'.format(t))
				g_t_probs = g.prob(X, obs[:,t], name = 'g_{}_probs'.format(t))

			W =  tf.divide(g_t_probs * f_t_probs, k * q_t_probs, name = 'W_{}'.format(t))	# (n_particles, batch_size)
			log_ZSMC += tf.log(tf.reduce_mean(W, axis = 0), name = 'log_ZSMC_{}'.format(t)) # (batch_size,)

			Xs.append(X)
			Ws.append(W)
			W_means.append(tf.reduce_mean(W, axis = 0))
			fs.append(f_t_probs)
			gs.append(g_t_probs)
			qs.append(q_t_probs)
			ps.append(k)

			mean_log_ZSMC = tf.reduce_mean(log_ZSMC)

		return mean_log_ZSMC, [Xs, Ws, W_means, fs, gs, qs, ps]

def tf_accuracy(obs_set, batch_size, log_ZSMC_args, debug_mode = False):
	len_obs_set = obs_set.shape.as_list()[0]
	VRNN_Cell, q, f, g, p, is_train, name = log_ZSMC_args

	accuracy = 0
	for i in range(0, len_obs_set, batch_size):
		obs = obs_set[i:i+batch_size]
		log_ZSMC, _ = get_log_ZSMC(obs, VRNN_Cell, q, f, g, p, is_train, name, debug_mode = debug_mode)
		accuracy += log_ZSMC
	return accuracy/(len_obs_set/batch_size)

def create_RLT_DIR(Experiment_params):
	Dh, time, n_particles, batch_size, lr, epoch, seed, n_train = Experiment_params
	# create the dir to save data
	cur_date = addDateTime()
	parser = OptionParser()

	parser.add_option("--rltdir", dest='rltdir', default='Experiment')
	args = sys.argv
	(options, args) = parser.parse_args(args)

	local_rlt_root = './rslts/SMC_LSTM/'
	params_str = "_Dh" + str(Dh) + "_T" + str(time) + "_n_particles" + str(n_particles) + \
				 "_batch_size" + str(batch_size) + "_lr" + str(lr) + \
				 "_epoch" + str(epoch) + "_n_train" + str(n_train) + \
				 "_seed" + str(seed)
	RLT_DIR = local_rlt_root + options.rltdir + params_str + cur_date + '/'

	if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)

	return RLT_DIR

if __name__ == '__main__':

	# hyperparameters

	Dy, Dx = 2, 2
	Dh = 100
	time = 25
	n_particles = 500

	batch_size = 5
	lr = 5e-4
	epoch = 100
	seed = 0

	n_train = 10 	* batch_size
	n_test  = 1 	* batch_size

	print_freq = 5

	debug_mode = True # if check shape in get_log_ZSMC

	tf.set_random_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	
	A_true = np.diag([0.95, 0.94])
	Q_true = np.asarray([[1., 0], [0, 1.]])
	B_true = np.diag([.95,.95])
	Sigma_true = np.asarray([[1., 0], [0, 1.]])
	x_0_true = np.array([1.0, 1.0])
	# x_0_true = np.random.randn(2)

	# whether create dir and store results (tf.graph, tf.summary, true A B Q x_0, optimized A B Q x_0)
	store_res = True
	if store_res == True:
		Experiment_params = (Dh, time, n_particles, batch_size, lr, epoch, seed, n_train)
		RLT_DIR = create_RLT_DIR(Experiment_params)

	# Create train and test dataset
	mySMC_sampler = SMC_sampler(Dx, Dy)
	f = multivariate_normal(A_true, Q_true, x_0_true)
	g = multivariate_normal(B_true, Sigma_true)

	hidden_train, obs_train = np.zeros((n_train, time, Dx)), np.zeros((n_train, time, Dy))
	hidden_test,  obs_test  = np.zeros((n_test,  time, Dx)), np.zeros((n_test,  time, Dy))
	for i in range(n_train + n_test):
		hidden, obs = mySMC_sampler.makePLDS(time, x_0_true, f, g)
		if i < n_train:
			hidden_train[i] = hidden
			obs_train[i]    = obs
		else:
			hidden_test[i-n_train] = hidden
			obs_test[i-n_train]    = obs
	obs_true = np.concatenate((obs_train, obs_test), axis = 0)
	print("finished creating dataset")

	# Plot training data
	plt.figure(figsize=(12,12))
	plt.title("Training Time Series")
	plt.xlabel("Time")
	for i in range(n_train):
		plt.subplot(8,7,i+1)
		plt.plot(hidden_train[i], c='red')
		plt.plot(obs_train[i], c='blue')
		sns.despine()
		plt.tight_layout()
	plt.savefig(RLT_DIR + "Training Data")
	plt.show()



	# plt.plot(hidden[:,0])
	# plt.plot(obs[:,0])
	# plt.savefig(RLT_DIR + "Training Data")
	# plt.show()

	# init A, B, Q, Sigma, x_0 randomly
	# A_init = np.random.rand(Dx, Dx)
	# B_init = np.random.rand(Dy, Dx)
	# L_Q_init = np.random.rand(Dx, Dx) # Q = L * L^T
	# L_Sigma_init = np.random.rand(Dy, Dy) # Q = L * L^T
	# x_0_init = np.random.rand(Dx)
	A_init = np.diag([0.9, 0.9])
	B_init = np.diag([0.9, 0.9])
	L_Q_init = np.asarray([[1., 0], [0, 1.]]) # Q = L * L^T
	L_Sigma_init = np.asarray([[1., 0], [0, 1.]]) # Q = L * L^T
	x_0_init = np.array([1.0, 1.0])

	print("A_init")
	print(A_init)
	print("B_init")
	print(B_init)
	print("Q_init")
	print(np.dot(L_Q_init, L_Q_init.T))
	print("Sigma_init")
	print(np.dot(L_Sigma_init, L_Sigma_init.T))
	print("x_0_init")
	print(x_0_init)

	# ================================ TF stuffs starts ================================ #

	# placeholders
	obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name = 'obs')
	obs_set_true  = tf.placeholder(tf.float32, shape=(n_train + n_test, time, Dy), 	name = 'obs_set_true')
	obs_set_train = tf.placeholder(tf.float32, shape=(n_train, 			time, Dy), 	name = 'obs_set_train')
	obs_set_test  = tf.placeholder(tf.float32, shape=(n_test, 			time, Dy), 	name = 'obs_set_test')

	VRNN_Cell = VartiationalRNN(Dx, Dy, Dh, n_particles, batch_size, debug_mode = debug_mode)

	# for evaluating true log_ZSMC
	A_true_tnsr 	= tf.Variable(A_true, 		dtype=tf.float32, name='A_true')
	B_true_tnsr 	= tf.Variable(B_true, 		dtype=tf.float32, name='B_true')
	Q_true_tnsr 	= tf.Variable(Q_true, 		dtype=tf.float32, name='Q_true')
	Sigma_true_tnsr = tf.Variable(Sigma_true, 	dtype=tf.float32, name='Q_true')
	x_0_true_tnsr 	= tf.Variable(x_0_true, 	dtype=tf.float32, name='x_0_true')
	q_true = tf_multivariate_normal(n_particles, batch_size, tf.eye(Dx),  tf.eye(Dx), 				  name = 'q_true')
	f_true = tf_multivariate_normal(n_particles, batch_size, A_true_tnsr, Q_true_tnsr, x_0_true_tnsr, name = 'f_true')
	g_true = tf_multivariate_normal(n_particles, batch_size, B_true_tnsr, Sigma_true_tnsr, 			  name = 'g_true')
	p_true = TensorGaussianPostApprox(A_true_tnsr, B_true_tnsr, Q_true_tnsr, Sigma_true_tnsr, name = 'p_true')

	# A, B, Q, x_0 to train
	A 		= tf.Variable(A_init, 		dtype=tf.float32, name='A')
	B 		= tf.Variable(B_init, 		dtype=tf.float32, name='B')
	L_Q 	= tf.Variable(L_Q_init, 	dtype=tf.float32, name='L_Q')
	L_Sigma = tf.Variable(L_Sigma_init, dtype=tf.float32, name='L_Sigma')
	x_0 	= tf.Variable(x_0_init, 	dtype=tf.float32, name='x_0')
	Q 		= tf.matmul(L_Q, 	 L_Q, 	  transpose_b = True, name = 'Q')
	Sigma 	= tf.matmul(L_Sigma, L_Sigma, transpose_b = True, name = 'Sigma')

	q_train = tf_multivariate_normal(n_particles, batch_size, tf.eye(Dx), tf.eye(Dx), name = 'q_train')
	f_train = tf_multivariate_normal(n_particles, batch_size, A, 		  Q, x_0, 	  name = 'f_train')
	g_train = tf_multivariate_normal(n_particles, batch_size, B, 		  Sigma, 	  name = 'g_train')
	p_train = TensorGaussianPostApprox(A, B, Q, Sigma, name = 'p_train') # Revisit me

	# for train_op
	log_ZSMC, log = get_log_ZSMC(obs, VRNN_Cell, q_train, f_train, g_train, p_train, 
								 is_train = True,  name = 'log_ZSMC_train', debug_mode = debug_mode)
	# for printing accuracy during training
	log_ZSMC_args_true = (VRNN_Cell, q_true,  f_true,  g_true,  p_true,  False, 'log_ZSMC_true')
	log_ZSMC_args_test = (VRNN_Cell, q_train, f_train, g_train, p_train, True,  'log_ZSMC_train')
	log_ZSMC_true  = tf_accuracy(obs_set_true,  batch_size, log_ZSMC_args_true, debug_mode = debug_mode)
	log_ZSMC_train = tf_accuracy(obs_set_train, batch_size, log_ZSMC_args_test, debug_mode = debug_mode)
	log_ZSMC_test  = tf_accuracy(obs_set_test,  batch_size, log_ZSMC_args_test, debug_mode = debug_mode)
	
	with tf.name_scope('train'):
		train_op = tf.train.GradientDescentOptimizer(lr).minimize(-log_ZSMC)

	# tf summary stuffs
	A_smry 		= tf.summary.histogram('A', A)
	B_smry 		= tf.summary.histogram('B', B)
	Q_smry 		= tf.summary.histogram('Q', Q)
	Sigma_smry 	= tf.summary.histogram('Sigma', Sigma)
	x_0_smry 	= tf.summary.histogram('x_0', x_0)

	log_ZSMC_true_smry 	= tf.summary.scalar('log_ZSMC_true', log_ZSMC_true)
	log_ZSMC_train_smry = tf.summary.scalar('log_ZSMC_train', log_ZSMC_train)
	log_ZSMC_test_smry 	= tf.summary.scalar('log_ZSMC_test', log_ZSMC_test)

	matrix_merged = tf.summary.merge([A_smry, B_smry, Q_smry, Sigma_smry, x_0_smry])

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

		log_ZSMC_true_val, summary = sess.run([log_ZSMC_true, log_ZSMC_true_smry], 
											  feed_dict = {obs_set_true: obs_true})
		print("true log_ZSMC: {:<10.4f}".format(log_ZSMC_true_val))
		if store_res == True:
			writer.add_summary(summary, 0)

		for i in range(epoch):
			# train A, B, Q, x_0 using each training sample
			#np.random.shuffle(obs_train)
			for j in range(0, len(obs_train), batch_size):
				sess.run(train_op, feed_dict={obs: obs_train[j:j+batch_size]})
				if store_res == True:
					summary = sess.run(matrix_merged)
					writer.add_summary(summary, i * len(obs_train) + j)

			# print training and testing loss
			if (i+1)%print_freq == 0:
				log_ZSMC_train_val, train_smry = sess.run([log_ZSMC_train, log_ZSMC_train_smry], 
														  feed_dict = {obs_set_train: obs_train})
				log_ZSMC_test_val,  test_smry  = sess.run([log_ZSMC_test, log_ZSMC_test_smry],  
														  feed_dict = {obs_set_test:  obs_test})
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"\
					.format(i+1, log_ZSMC_train_val, log_ZSMC_test_val))

				log_ZSMC_trains.append(log_ZSMC_train_val)
				log_ZSMC_tests.append(log_ZSMC_test_val)
				if store_res == True:
					writer.add_summary(train_smry, (i+1) * len(obs_train))
					writer.add_summary(test_smry,  (i+1) * len(obs_train))

		A_val = A.eval()
		B_val = B.eval()
		Q_val = Q.eval()
		x_0_val = x_0.eval()

		trained_log_ZSMC, trained_params = sess.run([log_ZSMC, log], feed_dict={obs: obs_train[:batch_size]})
		test_log_ZSMC, test_params = sess.run([log_ZSMC, log], feed_dict = {obs: obs_test[:batch_size]})

		TrainLatents = np.zeros((time, n_particles, batch_size, Dx))
		for i, x in enumerate(test_params[0]):
			TrainLatents[i] = x
		#Paths = np.mean(Latents, axis=1)
		plt.plot(TrainLatents[:,:,0,0], alpha=0.01, c='black')
		plt.plot(hidden_train[0,:,0], c='yellow')
		plt.savefig(RLT_DIR + 'train_path_filtered')
		plt.show()

		plt.figure(figsize=(10,10))
		for b in range(batch_size):
			plt.subplot(5, 5, b + 1)
			plt.plot(TrainLatents[:,:,b,0], alpha=0.01)
			plt.plot(hidden_train[b,:,0], c='yellow')
		plt.show()

		Latents = np.zeros((time, n_particles, batch_size, Dx))
		for i, x in enumerate(test_params[0]):
			Latents[i] = x
		#Paths = np.mean(Latents, axis=1)
		plt.plot(Latents[:,:,0,0], alpha=0.01, c='black')
		plt.plot(hidden_test[0,:,0], c='yellow')
		plt.savefig(RLT_DIR + 'test_path_filtered')
		plt.show()

		plt.figure(figsize=(10,10))
		for b in range(batch_size):
			plt.subplot(5, 5, b + 1)
			plt.plot(Latents[:,:,b,0], alpha=0.01)
			plt.plot(hidden_test[b,:,0], c='yellow')
		plt.show()



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
		params_dict = {"Dh": Dh, "T":time, "n_particles":n_particles, "batch_size":batch_size, "lr":lr,
				 	   "epoch":epoch, "n_train":n_train, "seed":seed}
		true_model_dict = {"A_true":A_true, "Q_true":Q_true, "B_true":B_true, "x_0_true":x_0_true}
		learned_model_dict = {"A_val":A_val, "Q_val":Q_val, "B_val":B_val, "x_0_val":x_0_val}
		log_ZSMC_dict = {"log_ZSMC_true_val":log_ZSMC_true_val, "log_ZSMC_trains": log_ZSMC_trains, 
						 "log_ZSMC_tests":log_ZSMC_tests}
		data_dict = {"params_dict":params_dict, "true_model_dict":true_model_dict, 
					 "learned_model_dict":learned_model_dict, "log_ZSMC_dict":log_ZSMC_dict}
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)

	plt.figure()
	plt.plot([log_ZSMC_true_val] * len(log_ZSMC_trains), '--')
	plt.plot(log_ZSMC_trains)
	plt.plot(log_ZSMC_tests)
	plt.legend(['true_log_ZSMC_val', 'log_ZSMC_train', 'log_ZSMC_test'])
	sns.despine()
	if store_res == True:
	  plt.savefig(RLT_DIR + "log_ZSMC_vals")
	plt.show()

	# plt.figure()
	# plt.plot(Particles[:,:,1].T, alpha=0.01, c='black')
	# plt.plot(hidden[:, 1], c='yellow')
	# sns.despine()
	# if store_res == True:
	#   plt.savefig(RLT_DIR + "Filtered Paths Dim 2")
	# plt.show()
 
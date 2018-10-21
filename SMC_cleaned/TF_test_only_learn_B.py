import numpy as np
import scipy as sp
import random
import math

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow_probability import distributions as tfd
import pdb

# import from files
from SMC_sampler import SMC_sampler
from distributions import multivariate_normal, poisson, tf_multivariate_normal, tf_poisson
from posterior_approx import LaplaceApprox, GaussianPostApprox, TensorLaplaceApprox, TensorGaussianPostApprox

# for data saving stuff
import sys
import pickle
import json
from jsontools import NumpyEncoder
import os
from datetime import datetime
from datetools import addDateTime
from optparse import OptionParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to reduce a lot of log about the device


print("Akwaaba!")
print(tf.__version__)

def get_log_ZSMC(obs, n_particles, x_0, q, f, g, p, use_stop_gradient, name = "get_log_ZSMC"):
	with tf.name_scope(name):

		Dx = x_0.get_shape().as_list()[0]
		T, Dy = obs.get_shape().as_list()
		
		Xs = []
		Ws = []
		W_means = []
		fs = []
		gs = []
		qs = []
		ps = []

		# T = 1
		X = q.sample(None, name = 'X0')
		q_uno_probs = q.prob(None, X, name = 'q_uno_probs')
		f_nu_probs  = f.prob(None, X, name = 'f_nu_probs')
		g_uno_probs = g.prob(X, obs[0], name = 'g_uno_probs')

		W = tf.multiply(g_uno_probs, f_nu_probs / q_uno_probs, name = 'W_0')
		log_ZSMC = tf.log(tf.reduce_mean(W, name = 'W_0_mean'), name = 'log_ZSMC_0')

		Xs.append(X)
		Ws.append(W)
		W_means.append(tf.reduce_mean(W))
		fs.append(f_nu_probs)
		gs.append(g_uno_probs)
		qs.append(q_uno_probs)
		ps.append(tf.zeros(n_particles))

		for t in range(1, T):

			# W_{t-1} = W_{t-1} * p(y_t | X_{t-1})
			# k = p.posterior(X, obs[t], name = 'p_{}'.format(t))
			k = tf.ones(n_particles, dtype = tf.float32)
			W = W * k

			categorical = tfd.Categorical(probs = W/tf.reduce_sum(W), name = 'Categorical_{}'.format(t))
			if use_stop_gradient:
				idx = tf.stop_gradient(categorical.sample(n_particles))
			else:
				idx = categorical.sample(n_particles)

			X_prev = tf.gather(X, idx, validate_indices = True)
		
			X = q.sample(X_prev, name = 'q_{}_sample'.format(t))
			q_t_probs = q.prob(X_prev, X, name = 'q_{}_probs'.format(t))
			f_t_probs = f.prob(X_prev, X, name = 'f_{}_probs'.format(t))
			g_t_probs = g.prob(X, obs[t], name = 'g_{}_probs'.format(t))

			W =  tf.divide(g_t_probs * f_t_probs, k * q_t_probs, name = 'W_{}'.format(t))
			log_ZSMC += tf.log(tf.reduce_mean(W), name = 'log_ZSMC_{}'.format(t))

			Xs.append(X)
			Ws.append(W)
			W_means.append(tf.reduce_mean(W))
			fs.append(f_t_probs)
			gs.append(g_t_probs)
			qs.append(q_t_probs)
			ps.append(k)

		return log_ZSMC, [Xs, Ws, W_means, fs, gs, qs, ps]

def evaluate_mean_log_ZSMC(log_ZSMC, D, obs_samples, sess, debug_mode = False):
	"""
	used for evaluating true_log_ZSMC, train_log_ZSMC, test_log_ZSMC
	"""

	Xs, Ws, W_means, fs, gs, qs, ps = D

	log_ZSMCs = []
	for obs_sample in obs_samples:
		log_ZSMC_val = sess.run(log_ZSMC, feed_dict={obs: obs_sample})

		if debug_mode:
			print("log_ZSMC_val", log_ZSMC_val)
			for X, W, W_mean, f, g, q, p in zip(Xs, Ws, W_means, fs, gs, qs, ps):
				X, W, W_mean, f, g, q, p = sess.run([X, W, W_mean, f, g, q, p], feed_dict={obs: obs_sample})
				idx = np.argsort(-W)[:10]
				print("W_mean", W_mean)
				print("W", W[idx])
				print("X", X[idx])
				print("f", f[idx])
				print("g", g[idx])
				print("q", q[idx])
				print("p", p[idx])

		log_ZSMCs.append(log_ZSMC_val)

	return np.mean(log_ZSMCs)


def create_RLT_DIR(Experiment_params):
	n_particles, n_iters, time, lr, epoch, seed, n_train, use_stop_gradient = Experiment_params
	# create the dir to save data
	cur_date = addDateTime()
	parser = OptionParser()

	parser.add_option("--rltdir", dest='rltdir', default='Experiment')
	args = sys.argv
	(options, args) = parser.parse_args(args)

	local_rlt_root = './rslts/Only_learn_B/'

	params_str = "_n_particles" + str(n_particles) + "_n_iters" + str(n_iters) + \
				 "_T" + str(time) + "_lr" + str(lr) + "_epoch" + str(epoch) + "_seed" + str(seed) +\
				 "_n_train" + str(n_train) 
	params_str += "_with_stop_gradient" if use_stop_gradient else "without_stop_gradient"

	RLT_DIR = local_rlt_root + options.rltdir + params_str + cur_date + '/'

	if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)

	return RLT_DIR

if __name__ == '__main__':

	# hyperparameters
	n_particles = 10000
	n_iters = 2         # num_iters for Laplace Approx
	time = 100
	lr = 1e-4
	epoch = 100
	seed = 0

	n_train = 100
	n_test = 5

	print_freq = 10
	use_stop_gradient = False

	tf.set_random_seed(seed)
	np.random.seed(seed)
	random.seed(seed)

	
	A_true = np.asarray([[0.99, -0.25], [-0.05, 0.54]])
	Q_true = np.asarray([[1., 0], [0, 1.]])
	B_true = np.diag([5.0, 15.0])
	Sigma_true = np.asarray([[1., 0], [0, 1.]])
	x_0_true = np.array([1.0, 1.0])
	# x_0_true = np.random.randn(2)

	# whether create dir and store results (tf.graph, tf.summary, true A B Q x_0, optimized A B Q x_0)
	store_res = True
	if store_res == True:
		Experiment_params = (n_particles, n_iters, time, lr, epoch, seed, n_train, use_stop_gradient)
		RLT_DIR = create_RLT_DIR(Experiment_params)

	Dy, Dx = B_true.shape

	# Create train and test dataset
	mySMC_sampler = SMC_sampler(Dx, Dy)
	f = multivariate_normal(A_true, Q_true, x_0_true)
	g = multivariate_normal(B_true, Sigma_true)

	hidden_train, obs_train = [], []
	hidden_test, obs_test = [], []
	for i in range(n_train + n_test):
		hidden, obs = mySMC_sampler.makePLDS(time, x_0_true, f, g)
		if i < n_train:
			hidden_train.append(hidden)
			obs_train.append(obs)
		else:
			hidden_test.append(hidden)
			obs_test.append(obs)
	print("finish creating dataset")
	
	# Plot training data
	if not os.path.exists(RLT_DIR+"/Training Data"): os.makedirs(RLT_DIR+"/Training Data")
	for i in range(n_train):
		plt.figure()
		plt.title("Training Time Series")
		plt.xlabel("Time")
		plt.plot(hidden_train[i], c='red')
		plt.plot(obs_train[i], c='blue')
		sns.despine()
		plt.savefig(RLT_DIR+"Training Data/{}".format(i))
		plt.close()

	# init A, B, Q, Sigma, x_0 randomly
	# A_init = np.random.rand(Dx, Dx)
	# B_init = np.random.rand(Dy, Dx)
	# L_Q_init = np.random.rand(Dx, Dx) # Q = L * L^T
	# L_Sigma_init = np.random.rand(Dy, Dy) # Q = L * L^T
	# x_0_init = np.random.rand(Dx)
	A_init = A_true #np.diag([0.9, 0.9])
	B_init = np.diag([10.0, 10.0]) # B_true
	L_Q_init = Q_true # np.asarray([[1.2, 0], [0, 1.2]]) # Q = L * L^T
	L_Sigma_init = Sigma_true # np.asarray([[1.2, 0], [0, 1.2]]) # Q = L * L^T
	x_0_init = x_0_true # np.array([0.8, 0.8])
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

	obs = tf.placeholder(tf.float32, shape=(time, Dy), name = 'obs')

	# for evaluating true log_ZSMC
	A_true_tnsr 	= tf.Variable(A_true, 		dtype=tf.float32, trainable = False, name='A_true')
	B_true_tnsr 	= tf.Variable(B_true, 		dtype=tf.float32, trainable = False, name='B_true')
	Q_true_tnsr 	= tf.Variable(Q_true, 		dtype=tf.float32, trainable = False, name='Q_true')
	Sigma_true_tnsr = tf.Variable(Sigma_true, 	dtype=tf.float32, trainable = False, name='Q_true')
	x_0_true_tnsr 	= tf.Variable(x_0_true, 	dtype=tf.float32, trainable = False, name='x_0_true')
	q_true = tf_multivariate_normal(n_particles, tf.eye(Dx),  (10**2)*tf.eye(Dx), 			name = 'q_true')
	f_true = tf_multivariate_normal(n_particles, A_true_tnsr, Q_true_tnsr, x_0_true_tnsr, 	name = 'f_true')
	g_true = tf_multivariate_normal(n_particles, B_true_tnsr, Sigma_true_tnsr, 			  	name = 'g_true')
	p_true = TensorGaussianPostApprox(A_true_tnsr, B_true_tnsr, Q_true_tnsr, Sigma_true_tnsr, name = 'p_true')

	# A, B, Q, x_0 to train
	A 		= tf.Variable(A_init, 		dtype=tf.float32, trainable = False, name='A', )
	B 		= tf.Variable(B_init, 		dtype=tf.float32, 					 name='B')
	L_Q 	= tf.Variable(L_Q_init, 	dtype=tf.float32, trainable = False, name='L_Q')
	L_Sigma = tf.Variable(L_Sigma_init, dtype=tf.float32, trainable = False, name='L_Sigma')
	x_0 	= tf.Variable(x_0_init, 	dtype=tf.float32, trainable = False, name='x_0')
	Q 		= tf.matmul(L_Q, 	 L_Q, 	  transpose_b = True, name = 'Q')
	Sigma 	= tf.matmul(L_Sigma, L_Sigma, transpose_b = True, name = 'Sigma')

	q_train = tf_multivariate_normal(n_particles, tf.eye(Dx), (10**2)*tf.eye(Dx), name = 'q_train')
	f_train = tf_multivariate_normal(n_particles, A, 		  Q, x_0, 	  name = 'f_train')
	g_train = tf_multivariate_normal(n_particles, B, 		  Sigma, 	  name = 'g_train')
	p_train = TensorGaussianPostApprox(A, B, Q, Sigma, name = 'p_train')

	# true_log_ZSMC: log_ZSMC generated from true A, B, Q, x_0
	log_ZSMC_true, log_true = get_log_ZSMC(obs, n_particles, x_0_true_tnsr, q_true,  f_true,  g_true,  p_true,  
										   use_stop_gradient, name = 'log_ZSMC_true')
	log_ZSMC, log		 	= get_log_ZSMC(obs, n_particles, x_0, 		    q_train, f_train, g_train, p_train, 
										   use_stop_gradient, name = 'log_ZSMC_train')
	with tf.name_scope('train'):
		train_op = tf.train.GradientDescentOptimizer(lr).minimize(-log_ZSMC)
				
	A_smry = tf.summary.histogram('A', A)
	B_smry = tf.summary.histogram('B', B)
	Q_smry = tf.summary.histogram('Q', Q)
	Sigma_smry = tf.summary.histogram('Sigma', Sigma)
	x_0_smry = tf.summary.histogram('x_0', x_0)

	merged = tf.summary.merge([A_smry, B_smry, Q_smry, Sigma_smry, x_0_smry])
	loss_merged = None

	# tf summary writer
	if store_res == True:
		writer = tf.summary.FileWriter(RLT_DIR)

	init = tf.global_variables_initializer()

	log_ZSMC_trains = []
	log_ZSMC_tests = []
	with tf.Session() as sess:

		sess.run(init)

		writer.add_graph(sess.graph)

		true_log_ZSMC_val = evaluate_mean_log_ZSMC(log_ZSMC_true, log_true, obs_train + obs_test, sess)
		print("true_log_ZSMC_val: {:<10.4f}".format(true_log_ZSMC_val))

		for i in range(epoch):
			# train A, B, Q, x_0 using each training sample
			# np.random.shuffle(obs_train)
			for j, obs_sample in enumerate(obs_train):
				_, summary = sess.run([train_op, merged], feed_dict={obs: obs_sample})
				writer.add_summary(summary, i * len(obs_train) + j)

			# print training and testing loss
			if (i+1)%print_freq == 0:
				log_ZSMC_train = evaluate_mean_log_ZSMC(log_ZSMC, log, obs_train, sess)
				log_ZSMC_test  = evaluate_mean_log_ZSMC(log_ZSMC, log, obs_test, sess)
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"\
					.format(i+1, log_ZSMC_train, log_ZSMC_test))
				log_ZSMC_trains.append(log_ZSMC_train)
				log_ZSMC_tests.append(log_ZSMC_test)

				print("B")
				print(B.eval())

		A_val = A.eval()
		B_val = B.eval()
		Q_val = Q.eval()
		Sigma_val = Sigma.eval()
		x_0_val = x_0.eval()

		Xs = log[0]
		Xs_val = np.zeros((n_train, time, n_particles, Dx))
		for i in range(n_train):
			for j, X in enumerate(Xs):
				X_val = sess.run(X, feed_dict = {obs:obs_train[i]})
				Xs_val[i, j] = X_val

	sess.close()

	# Plot learning results
	if not os.path.exists(RLT_DIR+"/Learning Results"): os.makedirs(RLT_DIR+"/Learning Results")
	for i in range(n_train):
		plt.figure()
		plt.title("hidden state 0")
		plt.xlabel("Time")
		plt.plot(np.average(Xs_val[i, :, :, 0], axis = 1), alpha = 0.5, c = 'black')
		plt.plot(hidden_train[i][:, 0], c='yellow')
		sns.despine()
		plt.savefig(RLT_DIR+"/Learning Results/h_0_{}".format(i))
		plt.close()

		plt.figure()
		plt.title("hidden state 1")
		plt.xlabel("Time")
		plt.plot(np.average(Xs_val[i, :, :, 1], axis = 1), alpha = 0.5, c = 'black')
		plt.plot(hidden_train[i][:, 1], c='yellow')
		sns.despine()
		plt.savefig(RLT_DIR+"/Learning Results/h_1_{}".format(i))
		plt.close()

	plt.figure()
	plt.plot([true_log_ZSMC_val] * len(log_ZSMC_trains))
	plt.plot(log_ZSMC_trains)
	plt.plot(log_ZSMC_tests)
	plt.legend(['true_log_ZSMC_val', 'log_ZSMC_trains', 'log_ZSMC_tests'])
	sns.despine()
	if store_res == True:
	  plt.savefig(RLT_DIR + "log_ZSMC")
	plt.show()

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


	if store_res == True:
		params_dict = {"n_particles":n_particles, "n_iters":n_iters, "time":time, "lr":lr, "epoch":epoch, \
					   "seed":seed, "n_train":n_train, "use_stop_gradient":use_stop_gradient}
		true_model_dict = { "A_true":A_true, "Q_true":Q_true, 
							"B_true":B_true, "x_0_true":x_0_true}
		learned_model_dict = {"A_val":A_val, "Q_val":Q_val, 
							  "B_val":B_val, "x_0_val":x_0_val}
		log_ZSMC_dict = {"true_log_ZSMC":true_log_ZSMC_val, "log_ZSMC_trains": log_ZSMC_trains, 
						 "log_ZSMC_tests":log_ZSMC_tests}
		data_dict = {"params_dict":params_dict, "true_model_dict":true_model_dict, 
					 "learned_model_dict":learned_model_dict, "log_ZSMC_dict":log_ZSMC_dict}
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)
		with open(RLT_DIR + 'data.json', 'w') as f:
			json.dump(data_dict, f, indent = 4, cls = NumpyEncoder)


	# plt.figure()
	# plt.plot(Particles[:,:,1].T, alpha=0.01, c='black')
	# plt.plot(hidden[:, 1], c='yellow')
	# sns.despine()
	# if store_res == True:
	#   plt.savefig(RLT_DIR + "Filtered Paths Dim 2")
	# plt.show()
 
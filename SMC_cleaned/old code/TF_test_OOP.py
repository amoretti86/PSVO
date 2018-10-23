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
import os
from datetime import datetime
from datetools import addDateTime
from optparse import OptionParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # to reduce a lot of log about the device


print("Akwaaba!")
print(tf.__version__)

def get_log_ZSMC(obs, X, x_0, q, f, g, p, name = "get_log_ZSMC"):
	with tf.name_scope(name):

		n_particles, T, Dx = X.get_shape().as_list()
		Dy = obs.get_shape().as_list()[1]
		
		Ws = []
		W_means = []
		fs = []
		gs = []
		qs = []
		ps = []

		# T = 1
		q_uno_probs = q.prob(None,   X[:,0], name = 'q_uno_probs')
		f_nu_probs  = f.prob(None,   X[:,0], name = 'f_nu_probs')
		g_uno_probs = g.prob(X[:,0], obs[0], name = 'g_uno_probs')

		W = tf.multiply(g_uno_probs, f_nu_probs / q_uno_probs, name = 'W_0')
		log_ZSMC = tf.log(tf.reduce_mean(W, name = 'W_0_mean'), name = 'log_ZSMC_0')

		Ws.append(W)
		W_means.append(tf.reduce_mean(W))
		fs.append(f_nu_probs)
		gs.append(g_uno_probs)
		qs.append(q_uno_probs)
		ps.append(tf.zeros(500))

		for t in range(1, T):

			# W_{t-1} = W_{t-1} * p(y_t | X_{t-1})
			k = p.posterior(X[:, t-1], obs[t], name = 'p_{}'.format(t))
		
			q_t_probs = q.prob(X[:,t-1], X[:,t], name = 'q_{}_probs'.format(t))
			f_t_probs = f.prob(X[:,t-1], X[:,t], name = 'f_{}_probs'.format(t))
			g_t_probs = g.prob(X[:,t],   obs[t], name = 'g_{}_probs'.format(t))

			W =  tf.divide(g_t_probs * f_t_probs, k * q_t_probs, name = 'W_{}'.format(t))
			log_ZSMC += tf.log(tf.reduce_mean(W), name = 'log_ZSMC_{}'.format(t))

			Ws.append(W)
			W_means.append(tf.reduce_mean(W))
			fs.append(f_t_probs)
			gs.append(g_t_probs)
			qs.append(q_t_probs)
			ps.append(k)

		return log_ZSMC, [Ws, W_means, fs, gs, qs, ps]

def generate_X_samples(obs_samples, sess, model_params, sample_params, debug_mode = False):
	A, B, Q, Sigma, x_0 = model_params
	n_particles, n_iters, use_log_prob = sample_params

	A_val = A.eval(session = sess)
	B_val = B.eval(session = sess)
	Q_val = Q.eval(session = sess)
	Sigma_val = Sigma.eval(session = sess)
	x_0_val = x_0.eval(session = sess)

	if debug_mode:
		print(A_val)
		print(B_val)
		print(Q_val)
		print(Sigma_val)
		print(x_0_val)

	Dy, Dx = B_val.shape
	q = multivariate_normal(np.eye(Dx), np.eye(Dx), np.zeros(Dx))
	f = multivariate_normal(A_val, Q_val, x_0_val)
	g = multivariate_normal(B_val, Sigma_val)
	p = GaussianPostApprox(A_val, B_val, Q_val, Sigma_val)

	mySMC_sampler = SMC_sampler(Dx, Dy)
	X_samples = []
	for obs_sample in obs_samples:
		_, X_sample, _, _ = mySMC_sampler.sample(obs_sample, n_particles, q, f, g, p, use_log_prob)
		X_samples.append(X_sample)
	return X_samples

def evaluate_mean_log_ZSMC(log_ZSMC, D, obs_samples, sess, model_params, sample_params, debug_mode = False):
	"""
	used for evaluating true_log_ZSMC, train_log_ZSMC, test_log_ZSMC

	model_params = A, B, Q, Sigma, x_0
	sample_params = n_particles, n_iters, use_log_prob
	"""
	X_samples = generate_X_samples(obs_samples, sess, model_params, sample_params)

	Ws, W_means, fs, gs, qs, ps = D

	log_ZSMCs = []
	for obs_sample, X_sample in zip(obs_samples, X_samples):
		log_ZSMC_val = sess.run(log_ZSMC, feed_dict={obs: obs_sample, X: X_sample})

		if debug_mode:
			print("log_ZSMC_val", log_ZSMC_val)
			for W, W_mean, f, g, q, p in zip(Ws, W_means, fs, gs, qs, ps):
				W, W_mean, f, g, q, p = sess.run([W, W_mean, f, g, q, p], feed_dict={obs: obs_sample, X: X_sample})
				idx = np.argsort(-W)[:10]
				print("W_mean", W_mean)
				print("W", W[idx])
				print("f", f[idx])
				print("g", g[idx])
				print("q", q[idx])
				print("p", p[idx])

		log_ZSMCs.append(log_ZSMC_val)

	return tf.reduce_mean(log_ZSMCs)


def create_RLT_DIR(Experiment_params):
	n_particles, n_iters, time, lr, epoch, seed = Experiment_params
	# create the dir to save data
	cur_date = addDateTime()
	parser = OptionParser()

	parser.add_option("--rltdir", dest='rltdir', default='Experiment')
	args = sys.argv
	(options, args) = parser.parse_args(args)

	local_rlt_root = './rslts/APF_SMC/'

	params_str = "_n_particles" + str(n_particles) + "_n_iters" + str(n_iters) + \
				 "_T" + str(time) + "_lr" + str(lr) + "_epoch" + str(epoch) + "_seed" + str(seed)

	RLT_DIR = local_rlt_root + options.rltdir + params_str + cur_date + '/'

	if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)

	return RLT_DIR

if __name__ == '__main__':

	# hyperparameters
	n_particles = 500
	n_iters = 2         # num_iters for Laplace Approx
	time = 5
	lr = 1e-4
	epoch = 100
	seed = 0

	n_train = 15
	n_test = 5

	use_log_prob = True # if SMC sampler uses log_prob

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
		Experiment_params = (n_particles, n_iters, time, lr, epoch, seed)
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
	X = tf.placeholder(tf.float32, shape=(n_particles, time, Dx), name = 'X')
	obs = tf.placeholder(tf.float32, shape=(time, Dx), name = 'obs')

	# for evaluating true log_ZSMC
	A_true_tnsr 	= tf.Variable(A_true, 		dtype=tf.float32, name='A_true')
	B_true_tnsr 	= tf.Variable(B_true, 		dtype=tf.float32, name='B_true')
	Q_true_tnsr 	= tf.Variable(Q_true, 		dtype=tf.float32, name='Q_true')
	Sigma_true_tnsr = tf.Variable(Sigma_true, 	dtype=tf.float32, name='Q_true')
	x_0_true_tnsr 	= tf.Variable(x_0_true, 	dtype=tf.float32, name='x_0_true')
	q_true = tf_multivariate_normal(n_particles, tf.eye(Dx),  tf.eye(Dx), 				  name = 'q_true')
	f_true = tf_multivariate_normal(n_particles, A_true_tnsr, Q_true_tnsr, x_0_true_tnsr, name = 'f_true')
	g_true = tf_multivariate_normal(n_particles, B_true_tnsr, Sigma_true_tnsr, 			  name = 'g_true')
	p_true = TensorGaussianPostApprox(A_true_tnsr, B_true_tnsr, Q_true_tnsr, Sigma_true_tnsr, name = 'p_true')

	# A, B, Q, x_0 to train
	A 		= tf.Variable(A_init, 		dtype=tf.float32, name='A')
	B 		= tf.Variable(B_init, 		dtype=tf.float32, name='B')
	L_Q 	= tf.Variable(L_Q_init, 	dtype=tf.float32, name='L_Q')
	L_Sigma = tf.Variable(L_Sigma_init, dtype=tf.float32, name='L_Sigma')
	x_0 	= tf.Variable(x_0_init, 	dtype=tf.float32, name='x_0')
	Q 		= tf.matmul(L_Q, 	 L_Q, 	  transpose_b = True, name = 'Q')
	Sigma 	= tf.matmul(L_Sigma, L_Sigma, transpose_b = True, name = 'Sigma')
	q_train = tf_multivariate_normal(n_particles, tf.eye(Dx), tf.eye(Dx), name = 'q_train')
	f_train = tf_multivariate_normal(n_particles, A, 		  Q, x_0, 	  name = 'f_train')
	g_train = tf_multivariate_normal(n_particles, B, 		  Sigma, 	  name = 'g_train')
	p_train = TensorGaussianPostApprox(A, B, Q, Sigma, name = 'p_train')

	# true_log_ZSMC: log_ZSMC generated from true A, B, Q, x_0
	log_ZSMC_true, log_true = get_log_ZSMC(obs, X, x_0_true_tnsr, q_true,  f_true,  g_true,  p_true,  
										   name = 'log_ZSMC_true')
	log_ZSMC, log		 	= get_log_ZSMC(obs, X, x_0, 		  q_train, f_train, g_train, p_train, 
										   name = 'log_ZSMC_train')
	
	with tf.name_scope('train'):
		train_op = tf.train.AdamOptimizer(lr).minimize(-log_ZSMC)

	# store A, B, Q, x_0 and their gradients
	g_A, g_B, g_Q, g_x_0 = tf.gradients(log_ZSMC, [A, B, Q, x_0])
				
	A_smry = tf.summary.histogram('A', A)
	B_smry = tf.summary.histogram('B', B)
	Q_smry = tf.summary.histogram('Q', Q)
	Sigma_smry = tf.summary.histogram('Sigma', Sigma)
	x_0_smry = tf.summary.histogram('x_0', x_0)

	g_A_smry = tf.summary.histogram('g_A', g_A)
	g_B_smry = tf.summary.histogram('g_B', g_B)
	g_Q_smry = tf.summary.histogram('g_Q', g_Q)
	g_x_0_smry = tf.summary.histogram('g_x_0', g_x_0)

	merged = tf.summary.merge([A_smry, B_smry, Q_smry, x_0_smry, g_A_smry, g_B_smry, g_Q_smry, g_x_0_smry])
	merged = tf.summary.merge([A_smry, B_smry, Q_smry, Sigma_smry, x_0_smry])
	loss_merged = None

	# tf summary writer
	if store_res == True:
		writer = tf.summary.FileWriter(RLT_DIR)

	init = tf.global_variables_initializer()
	with tf.Session() as sess:

		sess.run(init)

		writer.add_graph(sess.graph)

		true_model_params = (A_true_tnsr, B_true_tnsr, Q_true_tnsr, Sigma_true_tnsr, x_0_true_tnsr)
		train_model_params = (A, B, Q, Sigma, x_0)
		sample_params = (n_particles, n_iters, use_log_prob)

		true_log_ZSMC_val = evaluate_mean_log_ZSMC(log_ZSMC_true, log_true, obs_train[:5] + obs_test, sess, 
												   true_model_params, sample_params)
		true_log_ZSMC_smry = tf.summary.scalar('true_log_ZSMC_val', true_log_ZSMC_val)
		print("true_log_ZSMC_val: {:<10.4f}".format(true_log_ZSMC_val.eval()))

		for i in range(epoch):
			# train A, B, Q, x_0 using each training sample
			np.random.shuffle(obs_train)
			for j, obs_sample in enumerate(obs_train):
				X_sample = generate_X_samples([obs_sample], sess, train_model_params, sample_params)[0]
				_, summary = sess.run([train_op, merged], feed_dict={obs: obs_sample, X: X_sample})
				writer.add_summary(summary, i * len(obs_train) + j)

			# print training and testing loss
			if (i+1)%1 == 0:
				log_ZSMC_train = evaluate_mean_log_ZSMC(log_ZSMC, log, obs_train[:5], 
														sess, train_model_params, sample_params)
				log_ZSMC_test  = evaluate_mean_log_ZSMC(log_ZSMC, log, obs_test, 
														sess, train_model_params, sample_params)
				log_ZSMC_train_smry = tf.summary.scalar('log_ZSMC_train', log_ZSMC_train)
				log_ZSMC_test_smry = tf.summary.scalar('log_ZSMC_test', log_ZSMC_test)
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}"\
					.format(i+1, log_ZSMC_train.eval(), log_ZSMC_test.eval()))
				
				if loss_merged is None:
					loss_merged = tf.summary.merge([true_log_ZSMC_smry, log_ZSMC_train_smry, log_ZSMC_test_smry])
				loss_summary = sess.run(loss_merged)
				writer.add_summary(loss_summary, i)

		A_val = A.eval()
		B_val = B.eval()
		Q_val = Q.eval()
		x_0_val = x_0.eval()


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
		params_dict = {"n_particles":n_particles, "n_iters":n_iters, "time":time, "lr":lr, "epoch":epoch, "seed":seed}
		true_model_dict = {"A_true":A_true, "Q_true":Q_true, "B_true":B_true, "x_0_true":x_0_true}
		learned_model_dict = {"A_val":A_val, "Q_val":Q_val, "B_val":B_val, "x_0_val":x_0_val}
		data_dict = {"params_dict":params_dict, "true_model_dict":true_model_dict, "learned_model_dict":learned_model_dict}
		with open(RLT_DIR + 'data.p', 'wb') as f:
			pickle.dump(data_dict, f)

	# plt.figure()
	# plt.plot(Particles[:,:,0].T, alpha=0.01, c='black')
	# plt.plot(hidden[:, 0], c='yellow')
	# sns.despine()
	# if store_res == True:
	#   plt.savefig(RLT_DIR + "Filtered Paths Dim 1")
	# plt.show()

	# plt.figure()
	# plt.plot(Particles[:,:,1].T, alpha=0.01, c='black')
	# plt.plot(hidden[:, 1], c='yellow')
	# sns.despine()
	# if store_res == True:
	#   plt.savefig(RLT_DIR + "Filtered Paths Dim 2")
	# plt.show()
 
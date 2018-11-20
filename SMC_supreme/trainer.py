import numpy as np
import math
from sklearn.utils import shuffle

import tensorflow as tf
import os

class trainer:
	def __init__(self,
				 Dx, Dy,
				 n_particles, time,
				 batch_size, lr, epoch,
				 MSE_steps,
				 store_res):
		self.Dx = Dx
		self.Dy = Dy

		self.n_particles = n_particles
		self.time = time

		self.batch_size = batch_size
		self.lr = lr
		self.epoch = epoch

		self.MSE_steps = MSE_steps

		self.store_res = store_res
		self.draw_quiver_during_training = False

	def set_rslt_saving(self, RLT_DIR, save_freq, saving_num):
		self.RLT_DIR = RLT_DIR
		self.save_freq = save_freq
		self.saving_num = saving_num
		self.writer = tf.summary.FileWriter(RLT_DIR)

	def set_SMC(self, SMC_true, SMC_train):
		self.SMC_true = SMC_true
		self.SMC_train = SMC_train

	def set_placeholders(self, x_0, obs, hidden):
		self.x_0 = x_0
		self.obs = obs
		self.hidden = hidden

	def set_quiver_arg(self, nextX, lattice):
		self.nextX = nextX
		self.lattice = lattice
		self.draw_quiver_during_training = True

	def evaluate(self, fetches, feed_dict_w_batches={}, average = False):
		"""
		fetches: a single tensor or list of tensor to evaluate
		feed_dict: {placeholder:input of multiple batches}
		"""
		if not feed_dict_w_batches:
			return self.sess.run(fetches)

		n_batches = len(list(feed_dict_w_batches.values())[0])
		assert n_batches%self.batch_size==0

		fetches_list = []
		feed_dict = {}
		for i in range(0, n_batches, self.batch_size):
			for key, value in feed_dict_w_batches.items():
				feed_dict[key] = value[i:i+self.batch_size]
			fetches_val = self.sess.run(fetches, feed_dict=feed_dict)
			fetches_list.append(fetches_val)

		res = []
		if isinstance(fetches, list):
			for i in range(len(fetches)):
				if isinstance(fetches_list[0][i], np.ndarray):
					tmp = np.concatenate([x[i] for x in fetches_list])
				else:
					tmp = np.array([x[i] for x in fetches_list])
				res.append(tmp)
		else:
			if isinstance(fetches_list[0], np.ndarray):
				res = np.concatenate(fetches_list)
			else:
				res = np.array(fetches_list)

		if average:
			if isinstance(res, list):
				res = [np.mean(x, axis = 0) for x in res]
			else:
				res = np.mean(res, axis = 0)

		return res

	def train(self, hidden_train, obs_train, hidden_test, obs_test, print_freq):
		log_ZSMC_true,  log_true  = self.SMC_true.get_log_ZSMC(self.obs, self.x_0, self.hidden)
		log_ZSMC_train, log_train = self.SMC_train.get_log_ZSMC(self.obs, self.x_0, self.hidden)

		MSE_true, _, _       = self.SMC_true.n_step_MSE(self.MSE_steps, self.hidden, self.obs)
		MSE_train, ys_hat, _ = self.SMC_train.n_step_MSE(self.MSE_steps, self.hidden, self.obs)
		
		with tf.variable_scope("train"):
			train_op = tf.train.AdamOptimizer(self.lr).minimize(-log_ZSMC_train)	

		init = tf.global_variables_initializer()

		if self.store_res == True:
			saver = tf.train.Saver(max_to_keep=1)

			log_ZSMC_trains = []
			log_ZSMC_tests = []
			MSE_trains = []
			MSE_tests = []

		self.sess = tf.Session()

		self.sess.run(init)

		if self.store_res == True:
			self.writer.add_graph(self.sess.graph)

		obs_all = np.concatenate((obs_train, obs_test))
		hidden_all = np.concatenate((hidden_train, hidden_test))

		# for tnsr in tf.trainable_variables():
		# 	print(tnsr)

		log_ZSMC_true_val = 0
		"""
		log_ZSMC_true_val = self.evaluate(log_ZSMC_true,
										  {self.obs:obs_all, self.x_0:hidden_all[:, 0], self.hidden:hidden_all},
										  average = True)
		"""
		MSE_true_val = self.evaluate(MSE_true,
									 {self.obs:obs_all, self.hidden:hidden_all},
									 average = True)

		print("true log_ZSMC: {:<7.3f}, true MSE: {:>7.3f}".format(log_ZSMC_true_val, MSE_true_val))

		log_ZSMC_train_val = self.evaluate(log_ZSMC_train,
										   {self.obs:obs_train, self.x_0:hidden_train[:, 0], self.hidden:hidden_train},
										   average = True)
		log_ZSMC_test_val = self.evaluate(log_ZSMC_train,
										  {self.obs:obs_test,  self.x_0:hidden_test[:, 0], self.hidden:hidden_test},
										  average = True)
		MSE_train_val = self.evaluate(MSE_train,
									  {self.obs:obs_train, self.hidden:hidden_train},
									  average = True)
		MSE_test_val  = self.evaluate(MSE_train,
									  {self.obs:obs_test,  self.hidden:hidden_test},
									  average = True)
		print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}, train MSE: {:>7.3f}, test MSE: {:>7.3f}"\
			.format(0, log_ZSMC_train_val, log_ZSMC_test_val, MSE_train_val, MSE_test_val))


		if self.store_res == True:
			log_ZSMC_trains.append(log_ZSMC_train_val)
			log_ZSMC_tests.append(log_ZSMC_test_val)
			MSE_trains.append(MSE_train_val)
			MSE_tests.append(MSE_test_val)

		for i in range(self.epoch):
			# train A, B, Q, x_0 using each training sample
			obs_train, hidden_train = shuffle(obs_train, hidden_train)
			for j in range(0, len(obs_train), self.batch_size):
				print(i, j)
				self.sess.run(train_op, feed_dict={self.obs:obs_train[j:j+self.batch_size], 
											  	   self.x_0:hidden_train[j:j+self.batch_size, 0],
											  	   self.hidden:hidden_train[j:j+self.batch_size]})

			# print training and testing loss
			if (i+1)%print_freq == 0:
				log_ZSMC_train_val = self.evaluate(log_ZSMC_train,
												   {self.obs:obs_train,
												    self.x_0:hidden_train[:, 0],
												    self.hidden:hidden_train},
												   average = True)
				log_ZSMC_test_val = self.evaluate(log_ZSMC_train,
												  {self.obs:obs_test,
												   self.x_0:hidden_test[:, 0],
												   self.hidden:hidden_test},
												  average = True)
				MSE_train_val = self.evaluate(MSE_train,
											  {self.obs:obs_train, self.hidden:hidden_train},
											  average = True)
				MSE_test_val  = self.evaluate(MSE_train,
											  {self.obs:obs_test, self.hidden:hidden_test},
											  average = True)
				print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}, train MSE: {:>7.3f}, test MSE: {:>7.3f}"\
					.format(i+1, log_ZSMC_train_val, log_ZSMC_test_val, MSE_train_val, MSE_test_val))

				if self.store_res == True:
					log_ZSMC_trains.append(log_ZSMC_train_val)
					log_ZSMC_tests.append(log_ZSMC_test_val)
					MSE_trains.append(MSE_train_val)
					MSE_tests.append(MSE_test_val)

				if self.draw_quiver_during_training == True:
					Xs = log_train[0]
					Xs_val = self.evaluate(Xs, {self.obs:obs_train[0:self.saving_num], 
												self.x_0:hidden_train[0:self.saving_num, 0],
												self.hidden:hidden_train[0:self.saving_num]})
					self.get_quiver_plot(Xs_val, self.nextX, self.lattice, i+1)

			if self.store_res == True and (i+1)%self.save_freq == 0:
				if not os.path.exists(self.RLT_DIR+"model/"): os.makedirs(self.RLT_DIR+"model/")
				saver.save(self.sess, self.RLT_DIR+"model/model_epoch", global_step=i+1)

		print("finish training")

		losses = [log_ZSMC_true_val, log_ZSMC_trains, log_ZSMC_tests, MSE_true_val, MSE_trains, MSE_tests]
		# sorry for the terrible name
		tensors = [log_true, log_train, ys_hat]

		return losses, tensors

	def close_session(self):
		self.sess.close()


	def get_quiver_plot(self, Xs_val, nextX, lattice, epoch):
		# Xs_val.shape = (batch_size, time, n_particles, Dx)
		X_trajs = np.mean(Xs_val, axis = 2)

		import matplotlib.pyplot as plt
		import seaborn as sns
		plt.figure()
		for X_traj in X_trajs:
			plt.plot(X_traj[:, 0], X_traj[:, 1])
			plt.scatter(X_traj[0, 0], X_traj[0, 1])
		plt.title("quiver")
		plt.xlabel("x_dim 1")
		plt.ylabel("x_dim 2")
		axes = plt.gca()
		x1range, x2range = axes.get_xlim(), axes.get_ylim()

		lattice_val = self.define2Dlattice(x1range, x2range)

		X = lattice_val
		nextX = self.sess.run(nextX, feed_dict={lattice:lattice_val})

		scale = int(5*max(abs(x1range[0]) + abs(x1range[1]), abs(x2range[0]) + abs(x2range[1]))/3)
		plt.quiver(X[:,:,0], X[:,:,1], nextX[:,:,0]-X[:,:,0], nextX[:,:,1]-X[:,:,1], scale=scale)

		sns.despine()
		if not os.path.exists(self.RLT_DIR+"quiver/"): os.makedirs(self.RLT_DIR+"quiver/")
		plt.savefig(self.RLT_DIR+"quiver/{}".format(epoch))
		plt.close()

	@staticmethod
	def define2Dlattice(x1range=(-30.0, 30.0), x2range=(-30.0, 30.)):

		x1coords = np.linspace(x1range[0], x1range[1])
		x2coords = np.linspace(x2range[0], x2range[1])
		Xlattice = np.stack(np.meshgrid(x1coords, x2coords), axis = 2)
		return Xlattice
import numpy as np
import scipy as sp
import random
import math
from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import pdb

# import from files
from SMC_sampler import SMC_sampler
from distributions import multivariate_normal, poisson, tf_multivariate_normal, tf_poisson
from SequentialMonteCarlo import *



# hyperparameters
n_particles = 10000
n_iters = 2  # num_iters for Laplace Approx
time = 50
lr = 1e-4
epoch = 100
seed = 0

n_train = 80
n_test = 5

print_freq = 5
use_stop_gradient = False
PlotSyntheticData = False#True

tf.set_random_seed(seed)
np.random.seed(seed)
random.seed(seed)


# True Parameters
A_true = np.asarray([[0.99, -0.25], [-0.05, 0.54]])
Q_true = np.asarray([[1., 0], [0, 1.]])
B_true = np.diag([5.0, 15.0])
Sigma_true = np.asarray([[1., 0], [0, 1.]])
x_0_true = np.array([1.0, 1.0])
# x_0_true = np.random.randn(2)


Dy, Dx = B_true.shape

# Create train and test dataset
mySMC_sampler = SMC_sampler(Dx, Dy)
f = multivariate_normal(A_true, Q_true, x_0_true)
g = multivariate_normal(B_true, Sigma_true)

hidden_train, obs_train = [], []
hidden_test, obs_test = [], []
print("Generating Synthetic Data...")
for i in range(n_train + n_test):
    hidden, obs = mySMC_sampler.makePLDS(time, x_0_true, f, g)
    if i < n_train:
        hidden_train.append(hidden)
        obs_train.append(obs)
    else:
        hidden_test.append(hidden)
        obs_test.append(obs)
print("Finished Creating Dataset :)")


# Plot Training Data
if PlotSyntheticData:
    print("Plotting data...")
    plt.figure(figsize=(12, 12))
    plt.title("Training Time Series")
    batch_size = 5
    for i in range(n_train):
        plt.subplot(n_train / batch_size, batch_size, i + 1)
        plt.plot(hidden_train[i], c='red')
        plt.plot(obs_train[i], c='blue')
        sns.despine()
        plt.tight_layout()
    plt.savefig("Training Data")
    plt.show()



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
#p_true = TensorGaussianPostApprox(A_true_tnsr, B_true_tnsr, Q_true_tnsr, Sigma_true_tnsr, name = 'p_true')

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
#p_train = TensorGaussianPostApprox(A, B, Q, Sigma, name = 'p_train')


# Define SMC objects for true model and training model
trueSMC  = SequentialMonteCarlo(obs, n_particles, x_0_true_tnsr, q_true, f_true, g_true, 1, name='logZSMCTrue')
trainSMC = SequentialMonteCarlo(obs, n_particles, x_0, q_train, f_train, g_train, 1, name='logZSMCtrain')


print("it worked, bitch.")

# Define operations as outputs of SMC functions
log_ZSMC_true, params_true = trueSMC.get_log_ZSMC(obs, use_stop_gradient=True, name='log_ZSMC_true')
log_ZSMC_train, params_train = trainSMC.get_log_ZSMC(obs, use_stop_gradient=True, name='log_ZSMC_train')




################# HERE IS THE IMPORTANT PART OF THE CODE #####################






with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(lr).minimize(-log_ZSMC_train)

A_smry = tf.summary.histogram('A', A)
B_smry = tf.summary.histogram('B', B)
Q_smry = tf.summary.histogram('Q', Q)
Sigma_smry = tf.summary.histogram('Sigma', Sigma)
x_0_smry = tf.summary.histogram('x_0', x_0)

merged = tf.summary.merge([A_smry, B_smry, Q_smry, Sigma_smry, x_0_smry])
loss_merged = None

# tf summary writer
#if store_res == True:
#    writer = tf.summary.FileWriter(RLT_DIR)

init = tf.global_variables_initializer()

log_ZSMC_trains = []
log_ZSMC_tests = []






### WHAT THE FUCK IS THIS AND WHY DO WE NEED IT? ####

def evaluate_mean_log_ZSMC(log_ZSMC, D, obs_samples, sess, debug_mode = True):
	"""
	used for evaluating true_log_ZSMC, train_log_ZSMC, test_log_ZSMC
	"""

	Xs, Ws, W_means, fs, gs, qs, ps = D

	log_ZSMCs = []
	#pdb.set_trace()
	for obs_sample in obs_samples:
		log_ZSMC_val, _ = sess.run(log_ZSMC, feed_dict={obs: obs_sample})

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


#pdb.set_trace()

sess = tf.InteractiveSession()
sess.run(init)
sess.run([log_ZSMC_true], feed_dict={obs:obs_train[0]})
obs_samples = obs_train + obs_test
sess.run(log_ZSMC_true, feed_dict={obs: obs_samples[0]})


for i in range(epoch):
    print("epoch %i" %i)
    start_time = datetime.now()
    #print("time: ", i)
    for j, obs_sample in enumerate(obs_train):
        _, summary = sess.run([train_op, merged], feed_dict={obs: obs_sample})
        #print(_, summary)

        if (i + 1) % print_freq == 0:

            log_ZSMC_train_ = sess.run(log_ZSMC_train, feed_dict={obs: obs_sample})
            #evaluate_mean_log_ZSMC(log_ZSMC_train, params_train, obs_train, sess)
            log_ZSMC_test_ = sess.run(log_ZSMC_train, feed_dict={obs: obs_test[0]})
                #evaluate_mean_log_ZSMC(log_ZSMC_train, params_train, obs_test, sess)
            print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}" \
                  .format(i + 1, log_ZSMC_train_, log_ZSMC_test_))
            log_ZSMC_trains.append(log_ZSMC_train_)
            log_ZSMC_tests.append(log_ZSMC_test_)

    print("B")
    print(B.eval())
    print("A")
    print(A.eval())
    end_time = datetime.now()
    print("Duration:", (end_time - start_time))

print("Done training")
#pdb.set_trace()
A_val = A.eval()
B_val = B.eval()
Q_val = Q.eval()
Sigma_val = Sigma.eval()
x_0_val = x_0.eval()

print("Evaluating filtered paths")
Xs = params_train[0]
Xs_val = np.zeros((n_train, time, n_particles, Dx))
for i in range(n_train):
    print("trial %i" %i)
    for j, X in enumerate(Xs):
        X_val = sess.run(X, feed_dict={obs: obs_train[i]})
        Xs_val[i, j] = X_val

np.save("Filtered Paths2", Xs_val)
#pdb.set_trace()

if True:
    print("Plotting data...")
    plt.figure(figsize=(12, 12))
    plt.title("Filtered Time Series")
    batch_size = 5
    for i in range(n_train):
        print("Figure %i" %i)
        #plt.subplot(n_train / batch_size, batch_size, i + 1)
        plt.figure()
        plt.plot(np.mean(Xs_val[i,:,:,0],axis=1), c='black')
        plt.plot(hidden_train[i], c='yellow')
        sns.despine()
        plt.tight_layout()
        plt.savefig("Mean Filtered Path %i" %i)
    plt.show()

if True:
    print("Plotting data...")
    plt.figure(figsize=(12, 12))
    plt.title("Filtered Time Series")
    batch_size = 5
    for i in range(n_train):
        print("Figure %i" %i)
        plt.subplot(n_train / batch_size, batch_size, i + 1)
        plt.plot(np.mean(Xs_val[i,:,:,0],axis=1), 'o', alpha=0.01, c='black')
        plt.plot(hidden_train[i], c='yellow')
        sns.despine()
        plt.tight_layout()
    plt.savefig("Filtered Paths")
    plt.show()


#pdb.set_trace()
assert False

with tf.Session() as sess:
    sess.run(init)

    obs_samples = obs_train + obs_test
    sess.run(log_ZSMC_true, feed_dict={obs: obs_samples[0]})



    #writer.add_graph(sess.graph)

    #true_log_ZSMC_val = evaluate_mean_log_ZSMC(log_ZSMC_true, log_true, obs_train + obs_test, sess)
    #print("true_log_ZSMC_val: {:<10.4f}".format(true_log_ZSMC_val))

    for i in range(epoch):
        # train A, B, Q, x_0 using each training sample
        # np.random.shuffle(obs_train)
        for j, obs_sample in enumerate(obs_train):
            _, summary = sess.run([train_op, merged], feed_dict={obs: obs_sample})
            #writer.add_summary(summary, i * len(obs_train) + j)

        # print training and testing loss
        if (i + 1) % print_freq == 0:
            log_ZSMC_train_ = evaluate_mean_log_ZSMC(log_ZSMC_train, params_train, obs_train, sess)
            log_ZSMC_test_ = evaluate_mean_log_ZSMC(log_ZSMC_train, params_train, obs_test, sess)
            print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}" \
                  .format(i + 1, log_ZSMC_train_, log_ZSMC_test_))
            log_ZSMC_trains.append(log_ZSMC_train_)
            log_ZSMC_tests.append(log_ZSMC_test_)

            print("B")
            print(B.eval())

    A_val = A.eval()
    B_val = B.eval()
    Q_val = Q.eval()
    Sigma_val = Sigma.eval()
    x_0_val = x_0.eval()

    Xs = params_train[0]
    Xs_val = np.zeros((n_train, time, n_particles, Dx))
    for i in range(n_train):
        for j, X in enumerate(Xs):
            X_val = sess.run(X, feed_dict={obs: obs_train[i]})
            Xs_val[i, j] = X_val

sess.close()





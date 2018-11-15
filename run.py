import tensorflow as tf

import numpy as np
import pickle
import json
import sys
from datetime import datetime
from sklearn.utils import shuffle


import matplotlib.pyplot as plt
import seaborn as sns

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from MLP import MLP_mvn, MLP_poisson
from distributions import mvn, poisson, tf_mvn, tf_poisson
from SMC import SMC
from rslts_saving import create_RLT_DIR, NumpyEncoder, plot_training_data, plot_learning_results, plot_losses


plot_training_data = False

# Load Data from File
datadir = '/Users/antoniomoretti/Desktop/dhern-ts_wcommona-b4b1ad88b3aa/data/fitzhughnagumo/'


n_particles = 100
batch_size = 4
lr = 3e-3
epoch = 200
seed = 0

print_freq = 1
store_res = True
save_freq = 50
max_fig_num = 20
rslt_dir_name = 'time_invariant_MLP_fhn_1D_Dev'

# Parameters to initialize the covariance of the proposal distribution
ProposalCovInit = 5
ProposalCovMin = 5

# create dir to store results
if store_res == True:
    Experiment_params = {"n_particles": n_particles, "time": time, "batch_size": batch_size,
                         "lr": lr, "epoch": epoch, "seed": seed, "n_train": n_train,
                         "rslt_dir_name": rslt_dir_name, "QCovInit": ProposalCovInit,
                         "QCovMin": ProposalCovMin}
    print('Experiment_params')
    for key, val in Experiment_params.items():
        print('\t{}:{}'.format(key, val))
    RLT_DIR = create_RLT_DIR(Experiment_params)
    print("RLT_DIR:", RLT_DIR)



with open(datadir + "datadict", 'rb') as handle:
    data = pickle.load(handle, encoding='latin1')

obs_train = data['Ytrain']
obs_test = data['Yvalid']
hidden_train = data['Xtrue'][0:80]
hidden_test = data['Xtrue'][80:100]

Dy, Dx = obs_train.shape[2], hidden_train.shape[2]
time = obs_train.shape[1]
n_train = obs_train.shape[0]
n_test = obs_test.shape[0]


def f(Y, t, I, a, b, c):
    # Euler discretization
    y1, y2 = Y
    return [y1 - (y1 ** 3) / 3 - y2 + I, a * (b * y1 - c * y2)]
    # return [y1 - (y1**3)/3 - y2, 0.08*(y1 + 0.7 - 0.8*y2)+I]

if plot_training_data:
    # plot phase portraint of the ODE system
    y1 = np.linspace(-3.0, 3.0, 25)
    y2 = np.linspace(-3.0, 3.0, 25)
    Y1, Y2 = np.meshgrid(y1, y2)
    tau = 0
    u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)
    NI, NJ = Y1.shape

    for i in range(NI):
        for j in range(NJ):
            x = Y1[i, j]
            y = Y2[i, j]
            yprime = f([x, y], tau, I=1, a=1, b=.95, c=.05)
            u[i, j] = yprime[0]
            v[i, j] = yprime[1]

    plt.figure(figsize=(10, 10))
    Q = plt.quiver(Y1, Y2, u, v, color='black')
    for p in hidden_train:
        plt.plot(p[:, 0], p[:, 1])
        plt.scatter(p[0, 0], p[0, 1])
    plt.savefig(RLT_DIR + "Latent Paths XData.png")


    plt.figure(figsize=(12,12))
    plt.title("Training Time Series")
    plt.xlabel("Time")
    for i in range(20):
        plt.subplot(5,4,i+1)
        #plt.plot(hidden_train[i], c='red')
        plt.plot(obs_train[i])
    sns.despine()
    plt.tight_layout()


# TensorFlow
# placeholders
obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name='obs')
x_0 = tf.placeholder(tf.float32, shape=(batch_size, Dx), name='x_0')

q_train = MLP_mvn(Dx + Dy, Dx, n_particles, batch_size, sigma_init=ProposalCovInit, sigma_min=ProposalCovMin, name='q_train')
f_train = MLP_mvn(Dx, Dx, n_particles, batch_size, sigma_init=1, sigma_min=1, name='f_train')
g_train = MLP_mvn(Dx, Dy, n_particles, batch_size, sigma_init=1, sigma_min=1, name='g_train')

# for train_op
#SMC_true = SMC(q_true, f_true, g_true, n_particles, batch_size, name='log_ZSMC_true')
SMC_train = SMC(q_train, f_train, g_train, n_particles, batch_size, name='log_ZSMC_train')
#log_ZSMC_true, log_true = SMC_true.get_log_ZSMC(obs, x_0)
log_ZSMC_train, log_train = SMC_train.get_log_ZSMC(obs, x_0)


with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(lr).minimize(-log_ZSMC_train)

if store_res == True:
    writer = tf.summary.FileWriter(RLT_DIR)
    saver = tf.train.Saver()


init = tf.global_variables_initializer()

# for plotting
log_ZSMC_trains = []
log_ZSMC_tests = []

with tf.Session() as sess:
    sess.run(init)

    if store_res == True:
        writer.add_graph(sess.graph)

    # log_ZSMC_true_val = SMC_true.tf_accuracy(sess, log_ZSMC_true, obs, obs_train + obs_test, x_0,
    #                                         hidden_train + hidden_test)
    # print("log_ZSMC_true_val: {:<7.3f}".format(log_ZSMC_true_val))
    """
    log_ZSMC_train_val = SMC_train.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
    log_ZSMC_test_val = SMC_train.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
    print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}" \
          .format(0, log_ZSMC_train_val, log_ZSMC_test_val))
    log_ZSMC_trains.append(log_ZSMC_train_val)
    log_ZSMC_tests.append(log_ZSMC_test_val)
    """
    for i in range(epoch):
        start_time = datetime.now()
        print("Epoch ", i)
        # train A, B, Q, x_0 using each training sample
        obs_train, hidden_train = shuffle(obs_train, hidden_train)
        for j in range(0, len(obs_train), batch_size):
            sess.run(train_op, feed_dict={obs: obs_train[j:j + batch_size],
                                          x_0: [hidden[0] for hidden in hidden_train[j:j + batch_size]]})

        # print training and testing loss
        if (i + 1) % print_freq == 0:
            """
            log_ZSMC_train_val = SMC_train.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
            log_ZSMC_test_val = SMC_train.tf_accuracy(sess, log_ZSMC_train, obs, obs_train, x_0, hidden_train)
            print("iter {:>3}, train log_ZSMC: {:>7.3f}, test log_ZSMC: {:>7.3f}" \
                  .format(i + 1, log_ZSMC_train_val, log_ZSMC_test_val))

            log_ZSMC_trains.append(log_ZSMC_train_val)
            log_ZSMC_tests.append(log_ZSMC_test_val)
            """
            SMC_train.plot_flow(sess, log_train[0], obs, obs_train, x_0, hidden_train, i, RLT_DIR=RLT_DIR)

        if store_res == True and (i + 1) % save_freq == 0:
            saver.save(sess, os.path.join(RLT_DIR, 'model/model_epoch'), global_step=i + 1)

        end_time = datetime.now()
        print("Duration:", (end_time - start_time))


    print("Done Fitting. Evaluating Results")
    Xs = log_train[0]
    Xs_val = np.zeros((n_train, time, n_particles, Dx))
    for i in range(0, min(len(hidden_train), max_fig_num), batch_size):
        X_val = sess.run(Xs, feed_dict={obs: obs_train[i:i + batch_size],
                                        x_0: [hidden[0] for hidden in hidden_train[i:i + batch_size]]})
        for j in range(batch_size):
            Xs_val[i + j] = X_val[:, :, j, :]

sess.close()

print("All Done.")
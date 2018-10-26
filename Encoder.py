import numpy as np
import random
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow_probability import distributions
from tensorflow.contrib.layers import fully_connected

from SMC_sampler import *
from distributions import *


batch_size = 5

data_path = '/data/fitzhughnagumo/'


with open(data_path + "datadict", "rb") as input_file:
    data_dict = pickle.load(input_file, encoding='latin1') # Use if pickled in python 2

print("loaded data:\n", data_dict)

Ytrain = data_dict['Ytrain']
Xtrue = data_dict['Xtrue']
Yvalid = data_dict['Yvalid']

print("Shape of Ytrain:", Ytrain.shape)

print("Plotting loaded data...")
plt.figure(figsize=(12, 12))
plt.title("Training Time Series")
plt.xlabel("Time")
for i in range(Ytrain.shape[0]):
    plt.subplot(Ytrain.shape[0] / batch_size, batch_size, i + 1)
    plt.plot(Ytrain[i], c='red')
    plt.plot(Xtrue[i], c='blue')
    sns.despine()
    plt.tight_layout()
# plt.savefig(RLT_DIR + "Training Data")
plt.show()

# hyperparameters


Dy, Dx = 1, 2
Dh = 64
time = 200
n_particles = 1000

batch_size = 5
lr = 5e-4
epoch = 100
seed = 0

alpha = 0.1
n_train = Ytrain.shape[0]
n_test = Yvalid.shape[0]

print_freq = 5

debug_mode = True  # if check shape in get_log_ZSMC
generate_data = False
# whether create dir and store results (tf.graph, tf.summary, true A B Q x_0, optimized A B Q x_0)
store_res = False

if store_res == True:
    Experiment_params = (Dh, time, n_particles, batch_size, lr, epoch, seed, n_train)
    RLT_DIR = create_RLT_DIR(Experiment_params)

if generate_data:
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    A_true = np.diag([0.95, 0.94])
    Q_true = np.asarray([[1., 0], [0, 1.]])
    B_true = np.diag([.95, .95])
    Sigma_true = np.asarray([[1., 0], [0, 1.]])
    x_0_true = np.array([1.0, 1.0])


    # Create train and test dataset
    mySMC_sampler = SMC_sampler(Dx, Dy)
    f = multivariate_normal(A_true, Q_true, x_0_true)
    g = multivariate_normal(B_true, Sigma_true)

    hidden_train, obs_train = np.zeros((n_train, time, Dx)), np.zeros((n_train, time, Dy))
    hidden_test, obs_test = np.zeros((n_test, time, Dx)), np.zeros((n_test, time, Dy))
    for i in range(n_train + n_test):
        hidden, obs = mySMC_sampler.makePLDS(time, x_0_true, f, g)
        if i < n_train:
            hidden_train[i] = hidden
            obs_train[i] = obs
        else:
            hidden_test[i - n_train] = hidden
            obs_test[i - n_train] = obs
    obs_true = np.concatenate((obs_train, obs_test), axis=0)
    print("finished creating dataset")

    # Plot training data
    plt.figure(figsize=(12, 12))
    plt.title("Training Time Series")
    plt.xlabel("Time")
    for i in range(n_train):
        plt.subplot(n_train / batch_size, batch_size, i + 1)
        plt.plot(hidden_train[i], c='red')
        plt.plot(obs_train[i], c='blue')
        sns.despine()
        plt.tight_layout()
    # plt.savefig(RLT_DIR + "Training Data")
    plt.show()
else:
    print("Yeah buddy!")


### Define Encoder Network

# placeholders
obs = tf.placeholder(tf.float32, shape=(batch_size, time, Dy), name='obs')
obs_set_true = tf.placeholder(tf.float32, shape=(n_train + n_test, time, Dy), name='obs_set_true')
obs_set_train = tf.placeholder(tf.float32, shape=(n_train, time, Dy), name='obs_set_train')
obs_set_test = tf.placeholder(tf.float32, shape=(n_test, time, Dy), name='obs_set_test')

# obs_train.reshape(n_train*time,Dy)
YInput = tf.placeholder(tf.float32, shape=(batch_size*time,Dy),name='Yinput')


# Network params
EnHlayers = 60
EvHlayers = 200

### Define Deterministic Encoder Network ###
# Write a class for me ...
with tf.variable_scope('Encoder'):
    Enc_layer_1 = fully_connected(YInput, EnHlayers, activation_fn = tf.nn.softmax, scope = "layer1")
    Enc_layer_2 = fully_connected(Enc_layer_1, EnHlayers, activation_fn=tf.nn.softmax, scope="layer2")
    X_hat = fully_connected(Enc_layer_2, Dx, activation_fn=None, scope='/output')
    X_NxTxDz = tf.reshape(X_hat, [batch_size, time, Dx], name='X_hat')

### Define Deterministic Evolution Network
# Write a class for me ...
with tf.variable_scope('EvolutionNN'):
    Ev_layer_1 = fully_connected(X_NxTxDz, EvHlayers, activation_fn=tf.nn.softplus, scope="layer1")
    Ev_layer_2 = fully_connected(Ev_layer_1, EvHlayers, activation_fn=tf.nn.softplus, scope="layer2")
    output = fully_connected(Ev_layer_2, Dx*Dx, activation_fn=None, scope="output")
    B_NxTxDzxDz = tf.reshape(output, [batch_size, time, Dx, Dx], name='dynamics_B')
    B_NTxDzxDz = tf.reshape(output, [batch_size*time, Dx, Dx])
    A_NTxDzxDz = tf.eye(Dx) + alpha * B_NTxDzxDz
    A_NxTxDzxDz = tf.reshape(A_NTxDzxDz, [batch_size, time, Dx, Dx], name='dynamics_A')



print("It worked!")

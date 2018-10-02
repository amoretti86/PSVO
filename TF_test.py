import numpy as np
import scipy as sp
import random
import math

import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import pdb

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


def makePLDS(A,B,Q,T,x_0):
    """ creates PLDS with exp link function """
    Dx = A.shape[0]
    Dy = B.shape[0]
    X = np.zeros((T,Dx))
    Y = np.zeros((T,Dy))
    X[0] = x_0
    QChol = np.linalg.cholesky(Q)
    for i in range(1,T):
        X[i] = np.dot(A, X[i-1,:]) + np.dot(QChol, np.random.randn(1,Dx)[0])
        Y[i] = np.random.poisson(np.exp(np.dot(B,X[i])))
    return X, Y

def bivariateGaussHermite(xt, wt, mean, T, gfunc, XX, YY):
    """ Computes \int e^{-z^2}f(z)dz where z \in R^2 using roots of Hermite polynomial as nodes """
    mesh = np.array([XX.flatten(), YY.flatten()]).T
    grid_trans = np.sqrt(2)*np.dot(mesh, T) + mean
    geval = np.asarray([gfunc(xx) for xx in grid_trans]).reshape([len(xt), len(xt)])
    c = 1/(np.pi)
    y_marginal = np.zeros(len(xt))
    for idx in range(len(xt)):
        y_marginal[idx] = np.dot(geval[idx,:], wt)

    area = np.dot(y_marginal, wt) * c

    return area


def LaplaceApprox(A_DzxDz, Zprev_Dz, Q_DzxDz, Qinv_DzxDz, B_DyxDz, y_Dy, mu_Dz=None, niters=10):
    """ Computes Laplace Approximation to exp{ -1/2 (z_t - mu)^T Q^{-1}(z - mu) - <exp(Bz),1> + <Bz,y> - <ln(y!),1> } """
    ones_Dy = np.ones(B_DyxDz.shape[0])
    Z_Dz = Zprev_Dz
    mu_Dz = np.dot(A_DzxDz, Zprev_Dz)
    #pdb.set_trace()
    for i in range(niters):
        # Compute FPI for the mean
        BtOnes_Dz = np.dot(B_DyxDz.T, ones_Dy)
        #QBtOnes_Dz = np.dot(Q_DzxDz, BtOnes_Dz)
        BZ_Dy = np.dot(B_DyxDz, Z_Dz)
        expBZ_Dy = np.exp(BZ_Dy)
        #expBZOnes_1x1 = np.dot(expBZ_Dy, ones_Dy)
        BtexpBZ_Dz = np.dot(B_DyxDz.T, expBZ_Dy)
        BtY_Dz = np.dot(B_DyxDz.T, y_Dy)
        QBtY_Dz = np.dot(Q_DzxDz, BtY_Dz)
        #Z_Dz = - QBtOnes_Dz * BtexpBZ_Dz + QBtY_Dz + mu_Dz
        Z_Dz = - np.dot(Q_DzxDz, BtexpBZ_Dz) + QBtY_Dz + mu_Dz
        # print "iter %i, mean:\n"%i, Z_Dz
        # Compute FPI for the hessian
        expBZ_DyxDy = np.diag(np.exp(np.dot(B_DyxDz, Z_Dz)))
        BtexpBZ_DzxDy = np.dot(B_DyxDz.T, expBZ_DyxDy)
        BtexpBZB_DzxDz = np.dot(BtexpBZ_DzxDy, B_DyxDz)
        H_DzxDz = BtexpBZB_DzxDz + Qinv_DzxDz
        # print "iter %i, Hessian:\n" % i, H_DzxDz
    SqInvDet = ((1 / np.linalg.det(H_DzxDz)) ** (1. / 2))
    PiTerm = ((2 * np.pi) ** (Q_DzxDz.shape[0] / 2))
    Pstar = make_mvn_pdf(mu_Dz, Q_DzxDz)(Z_Dz) * make_poisson(y_Dy)(np.dot(B_DyxDz, Z_Dz))
    Ztilde_1x1 =  SqInvDet * PiTerm #* Pstar
    # print "area: ", Ztilde_1x1

    return Ztilde_1x1

def TensorLaplaceApprox(A_DzxDz,  Q_DzxDz, Qinv_DzxDz, B_DyxDz, Zprev_NxDz, y_Dy, ones_Dy, mu_Dz = None, niter=2, debug_mode=False):
    """
    Computes Laplace Approx using an FPI for the first and second moments differentiating the log posterior.
    Broadcasts computation to accept a rank 2 tensor as input for the latent states and compute N integrals
    of the form \hat p(y_t|x_{t-1}) = \int f(x_t|x_{t-1})g(y_t|x_t) dx_{t-1}
    """
    #pdb.set_trace()
    with tf.name_scope('TensorLaplaceApprox'):
        Z_NxDz = Zprev_NxDz
        mu_NxDz = tf.matmul(Zprev_NxDz, A_DzxDz)

        # Iterate over FPIs for first and second moments:
        for i in range(niter):
            # Compute FPI for the mean
            BtOnes_Dz = tf.matmul(ones_Dy,B_DyxDz)
            #QBtOnes_Dz = tf.matmul(BtOnes_Dz, Q_DzxDz)
            BZ_NxDy = tf.matmul(Z_NxDz, B_DyxDz, transpose_b=True)
            expBZ_NxDy = tf.exp(BZ_NxDy)
            #expBZOnes_Nx1 = tf.matmul(expBZ_NxDy, ones_Dy, transpose_b=True)
            BtexpBZ_NxDz = tf.matmul(expBZ_NxDy, B_DyxDz)
            BtY_Dz = tf.matmul(y_Dy, B_DyxDz)
            QBtY_Dz = tf.matmul(BtY_Dz, Q_DzxDz)
            QBtexpBZ_NxDz = tf.matmul(BtexpBZ_NxDz, Q_DzxDz)
            #Z_NxDz = - QBtOnes_Dz * expBZOnes_Nx1 + QBtY_Dz + mu_NxDz
            Z_NxDz = -QBtexpBZ_NxDz + QBtY_Dz + mu_NxDz
            if debug_mode:
                print("iter %i, mean:\n"%i, Z_NxDz)
            # Compute FPI for the Hessian
            expBZ_NxDyxDy = tf.matrix_diag(tf.exp((tf.matmul(Z_NxDz,B_DyxDz, transpose_b=True))))
            BtexpBZ_NxDzxDy = tf.einsum('ijk,jh->ihk', expBZ_NxDyxDy, B_DyxDz)
            BtexpBZB_NxDzxDz = tf.einsum('ijk,kh->ijh', BtexpBZ_NxDzxDy, B_DyxDz)
            H_NxDzxDz = BtexpBZB_NxDzxDz + Qinv_DzxDz
            if debug_mode:
                print("iter %i, Tensor of Hessians:\n", H_NxDzxDz)
        # Compute the inverse normalization to approximate the integral
        SqInvDet = ((1./tf.sqrt(tf.matrix_determinant(H_NxDzxDz)))**(1./2.))
        PiTerm = ((2*tf.constant(np.pi)))**(tf.shape(Qinv_DzxDz,out_type=tf.float32)[0]/2)
        #pdb.set_trace()
        # Ignore this term ...
        #Pstar  = make_mvn_pdf(mu_Dz, Q_DzxDz)(Z_NxDz)
        Ztilde_Nx1 = SqInvDet * PiTerm
        return Ztilde_Nx1

def make_mvn_pdf(mu, sigma):

    """ Define Gaussian density """
    def f(x):
        return sp.stats.multivariate_normal.pdf(x,mu,sigma)
    return f

def make_poisson(k):

    """ Define Poisson density with independent rates """
    def f(theta):
        prob = 1
        for i in range(len(k)):
            prob *= sp.stats.poisson.pmf(k[i], np.exp(theta[i]))
        return prob
    return f

def APF_SMC_sampler(obs, n_particles, n_iters, A, B, Q, x_0):
    """ Run SMC to sample latent paths (X) , weights (W), ancestor indices (a) and posterior integrals (k) """
    time = len(obs)

    # Initialize variables
    Dy, Dx = B.shape
    X = np.zeros((n_particles, time, Dx))
    a = np.zeros((n_particles, time))
    W = np.zeros((n_particles, time))
    k = np.zeros((n_particles, time))
    Qinv = np.linalg.inv(Q)
    n_gridpoints = 10
    [xt, wt] = np.polynomial.hermite.hermgauss(n_gridpoints)
    XX = np.tile(xt.reshape([1, len(xt)]), [len(xt), 1])
    YY = np.tile(xt.reshape([len(xt), 1]), [1, len(xt)])
    T = sp.linalg.sqrtm(Q)


    # Initialize particles and weights at time 1
    for i in range(n_particles):
        X[i, 0, :] = np.random.randn(1, Dx)[0]
        g = make_poisson(obs[0, :])(np.dot(B, X[i, 0, :]))
        nu = make_mvn_pdf(x_0, Q)(X[i, 0, :])
        q = make_mvn_pdf(np.zeros(Dx), np.eye(Dx))(X[i, 0, :])
        W[i, 0] = g * nu / q

    # Define recursive approximation to p(z_{1:n}|x_{1:n})
    for t in range(1, time):

        # Update weights and propagate particles based on posterior integral
        for i in range(n_particles):

            # Approximate posterior for the APF p(x_n|z_{n-1}) \propto f(z_n|z_{n-1})g(x_n|z_n)

            #k[i, t] = LaplaceApprox(A, X[i, t - 1, :], Q, Qinv, B, obs[t, :], n_iters)
            #print("Laplace particle {}, mass: {}".format(i, k[:, t]))

            # Quadrature can be used when Dz == Dy
            g_int_func = make_poisson(obs[t, :])
            k[i, t] = bivariateGaussHermite(xt, wt, np.dot(A, X[i, t - 1, :]), T, g_int_func, XX, YY)
            # print("Quadrature particle {}, mass: {}\n".format(i, k2[i, t]))
            #assert(math.isfinite(k[i, t])), 'LaplaceApprox generates invalid number {}'.format(k[i, t])
            #if not math.isfinite(k[i,t]):
            #    k[i,t] = 0

            # Reweight particles
            W[i, t - 1] = W[i, t - 1] * k[i, t]
        print("Laplace particle:\n", t, "\n", k[:, t])
        W[:, t - 1] = W[:, t - 1] / np.sum(W[:, t - 1])

        # Resample
        Xprime = np.random.choice(n_particles, n_particles, p = W[:, t - 1], replace = True)
        a[:, t] = Xprime
        Xtilde = [X[i, t - 1, :] for i in Xprime]

        # Reset weights and particles
        for i in range(n_particles):
            # Select new particles
            X[i, t - 1, :] = Xtilde[i]
            # Resample particles and reset weights
            X[i, t, :] = np.random.randn(1, Dx)[0] + X[i, t - 1, :]
            # Update factorized proposal and target distributions
            g = make_poisson(obs[t])(np.dot(B, X[i, t, :]))
            q = make_mvn_pdf(X[i, t - 1, :], np.eye(Dx))(X[i, t, :])
            f = make_mvn_pdf(np.dot(A, X[i, t - 1, :]), Q)(X[i, t, :])
            # Update weights
            W[i, t] = (g * f) / (k[i, t] * q)
        # print("time: %i" % t)

    X = X.astype(np.float32)
    return W, X, k, a

def get_log_Z_SMC(obs, X, A, B, Q, x_0, n_iters, name = "get_log_Z_SMC"):
    with tf.name_scope(name):
        n_particles = X.shape[0]
        Dy, Dx = B.shape
        T = obs.shape[0]

        ones_Dy = np.expand_dims(np.ones(obs.shape[1], dtype=np.float32), axis=0)
        Qinv = tf.matrix_inverse(Q, name = "Qinv")

        # T = 1
        q_uno = tfd.MultivariateNormalDiag(loc = np.zeros(Dx, dtype=np.float32), name = 'q_uno')
        q_uno_probs = q_uno.prob(X[:, 0, :], name = 'q_uno_probs')

        f_nu_uno = tfd.MultivariateNormalDiag(loc = x_0, name = 'f_nu_uno')
        f_nu_probs = f_nu_uno.prob(X[:, 0, :], name = 'f_nu_probs')

        g_uno = tfd.Poisson(log_rate=tf.matmul(B, X[:, 0, :], transpose_b=True, name = 'B_x_X_0T'), name = 'g_uno') 
        obs_uno = tf.tile(tf.expand_dims(obs[0,:], axis = 1), [1, n_particles], name = 'obs_uno')
        g_uno_probs = tf.reduce_prod(g_uno.prob(obs_uno), axis=(0), name = 'g_uno_probs')

        W = tf.multiply(g_uno_probs, f_nu_probs / q_uno_probs, name = 'W_0')

        log_ZSMC = tf.log(tf.reduce_mean(W, name = 'W_0_mean'), name = 'log_ZSMC_0')


        for t in range(1, T):
            # Broadcast particle operations at each time point
            # W_{t-1} = W_{t-1} * p(y_t | X_{t-1})
            obs_t = tf.expand_dims(obs[t,:], axis=0, name = 'obs_t_expaned')
            k = TensorLaplaceApprox(A, Q, Qinv, B, X[:, t-1, :], obs_t, ones_Dy, n_iters)
            W = W * k
        
            q_t = tfd.MultivariateNormalDiag(loc=X[:,t-1], scale_diag=np.ones(Dx, dtype=np.float32), name = 'q_{}'.format(t))
            q_t_probs = q_t.prob(X[:,t], name = 'q_{}_probs'.format(t))

            f_t = tfd.MultivariateNormalDiag(loc=X[:, t-1], scale_diag=np.ones(Dx, dtype=np.float32), name = 'f_{}'.format(t))
            f_t_probs = f_t.prob(X[:,t], name = 'f_{}_probs'.format(t))

            # Define emission density
            g_t = tfd.Poisson(log_rate=tf.matmul(B, X[:, t], transpose_b=True), name = 'g_{}'.format(t)) 
            obs_t = tf.tile(tf.expand_dims(obs[t,:], axis = 1), [1, n_particles], name = 'obs_t_tile')
            g_t_probs = tf.reduce_prod(g_t.prob(obs_t), axis=0, name = 'g_{}_probs'.format(t))

            W =  tf.divide(g_t_probs * f_t_probs[0], k * q_t_probs[0], name = 'W_{}'.format(t))
            log_ZSMC += tf.log(tf.reduce_mean(W), name = 'log_ZSMC_{}'.format(t))

        return log_ZSMC

def evaluate_mean_log_ZSMC(obs_set, log_ZSMC, A, B, Q, x_0, n_particles, n_iters):
    # used for evaluating train_log_ZSMC and test_log_ZSMC
    sess = tf.get_default_session()

    A_val = A.eval(session = sess)
    B_val = B.eval(session = sess)
    Q_val = Q.eval(session = sess)
    x_0_val = x_0.eval(session = sess)

    log_ZSMCs = []
    for obs_sample in obs_set:
        _, X_sample, _, _ = APF_SMC_sampler(obs_sample, n_particles, n_iters, A_val, B_val, Q_val, x_0_val)
    
        log_ZSMC_val = sess.run(log_ZSMC, feed_dict={obs: obs_sample, X: X_sample})
        log_ZSMCs.append(log_ZSMC_val)

    return np.mean(log_ZSMCs)


def create_RLT_DIR(n_particles, n_iters, time, lr, epoch, seed):
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
    epoch = 5000
    seed = 0

    n_train = 16
    n_test = 4

    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    
    A_true = np.diag([0.95, 0.94])
    Q_true = np.asarray([[1., 0], [0, 1.]])
    B_true = np.diag([.95,.95])
    #pdb.set_trace()
    x_0_true = np.array([1.0, 1.0])
    # x_0_true = np.random.randn(2)

    # whether create dir and store results (tf.graph, tf.summary, true A B Q x_0, optimized A B Q x_0)
    store_res = True
    if store_res == True:
        RLT_DIR = create_RLT_DIR(n_particles = 500, n_iters = 2, time = 3, lr = 1e-4, epoch = 100, seed = 1)

    Dy, Dx = B_true.shape

    # Create train and test dataset
    hidden_train, obs_train = [], []
    hidden_test, obs_test = [], []
    for i in range(n_train + n_test):
        hidden, obs = makePLDS(A_true, B_true, Q_true, time, x_0_true)
        if i < n_train:
            hidden_train.append(hidden)
            obs_train.append(obs)
        else:
            hidden_test.append(hidden)
            obs_test.append(obs)
    print("finish creating dataset")

    plt.plot(hidden_train[:,0])
    plt.plot(obs_train[:,0])
    plt.savefig(RLT_DIR + "Training Data")
    plt.show()

    # init A, B, Q, x_0 randomly
    A_init = np.diag([.9, .9])#np.random.rand(Dx, Dx)
    L_init = np.asarray([[1., 0.], [0., 1.]])#np.random.rand(Dx, Dx) # Q = L * L^T
    B_init = np.diag([.9, .9])#np.random.rand(Dy, Dx)
    x_0_init = np.array([1.0, 1.0])#np.random.rand(Dx)
    print("A_init")
    print(A_init)
    print("Q_init")
    print(np.dot(L_init, L_init.T))
    print("B_init")
    print(B_init)
    print("x_0_init")
    print(x_0_init)

    X = tf.placeholder(tf.float32, shape=(n_particles, time, Dx), name = 'X')
    obs = tf.placeholder(tf.float32, shape=(time, Dx), name = 'obs')

    # for evaluating true log_ZSMC
    A_true_tnsr = tf.Variable(A_true, dtype=tf.float32, name='A_true')
    B_true_tnsr = tf.Variable(B_true, dtype=tf.float32, name='B_true')
    Q_true_tnsr = tf.Variable(Q_true, dtype=tf.float32, name='Q_true')
    x_0_true_tnsr = tf.Variable(x_0_true, dtype=tf.float32, name='x_0_true')

    # A, B, Q, x_0 to train
    A = tf.Variable(A_init, dtype=tf.float32, name='A')
    B = tf.Variable(B_init, dtype=tf.float32, name='B')
    L = tf.Variable(L_init, dtype=tf.float32, name='Q')
    Q = tf.matmul(L, L, transpose_b = True)
    x_0 = tf.Variable(x_0_init, dtype=tf.float32, name='x_0')

    # true_log_ZSMC: log_ZSMC generated from true A, B, Q, x_0
    true_log_ZSMC = get_log_Z_SMC(obs, X, A_true_tnsr, B_true_tnsr, Q_true_tnsr, x_0_true_tnsr, n_iters, name = 'true_log_ZSMC')
    log_ZSMC = get_log_Z_SMC(obs, X, A, B, Q, x_0, n_iters, name = 'optimzing_log_ZSMC')

    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(lr).minimize(-log_ZSMC)

    # store A, B, Q, x_0 and their gradients
    g_A, g_B, g_Q, g_x_0 = tf.gradients(log_ZSMC, [A, B, Q, x_0])

    # Write to TensorBoard
    A_smry = tf.summary.histogram('A', A)
    B_smry = tf.summary.histogram('B', B)
    Q_smry = tf.summary.histogram('Q', Q)
    x_0_smry = tf.summary.histogram('x_0', x_0)

    # Write to TensorBoard
    g_A_smry = tf.summary.histogram('g_A', g_A)
    g_B_smry = tf.summary.histogram('g_B', g_B)
    g_Q_smry = tf.summary.histogram('g_Q', g_Q)
    g_x_0_smry = tf.summary.histogram('g_x_0', g_x_0)

    merged = tf.summary.merge([A_smry, B_smry, Q_smry, x_0_smry, g_A_smry, g_B_smry, g_Q_smry, g_x_0_smry])
    loss_merged = None

    # tf summary writer
    if store_res == True:
        writer = tf.summary.FileWriter(RLT_DIR)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        sess.run(init)

        writer.add_graph(sess.graph)

        true_log_ZSMC_val = evaluate_mean_log_ZSMC(obs_train[:5] + obs_test, true_log_ZSMC, 
                                                   A_true_tnsr, B_true_tnsr, Q_true_tnsr, x_0_true_tnsr, 
                                                   n_particles, n_iters)
        true_log_ZSMC_summary = tf.summary.scalar('true_log_ZSMC_val', true_log_ZSMC_val)
        print("true_log_ZSMC_val: {:<10.4f}".format(true_log_ZSMC_val))

        for i in range(epoch):
            # train A, B, Q, x_0 using each training sample
            np.random.shuffle(obs_train)
            for j, obs_sample in enumerate(obs_train):
                A_val = A.eval()
                B_val = B.eval()
                Q_val = Q.eval()
                x_0_val = x_0.eval()

                _, X_sample, _, _ = APF_SMC_sampler(obs_sample, n_particles, n_iters, A_val, B_val, Q_val, x_0_val)
                plt.plot(X_sample[:,:,0], alpha=0.01)
                plt.plot(hidden[:,0], c='yellow')
                plt.show()


                _, summary = sess.run([train_op, merged], feed_dict={obs: obs_sample, X: X_sample})
                writer.add_summary(summary, i * len(obs_train) + j)

            # print training and testing loss
            if (i+1)%5 == 0:
                log_ZSMC_train = evaluate_mean_log_ZSMC(obs_train[:5], log_ZSMC, A, B, Q, x_0, n_particles, n_iters)
                log_ZSMC_test = evaluate_mean_log_ZSMC(obs_test, log_ZSMC, A, B, Q, x_0, n_particles, n_iters)
                print("iter {:<6}, train log_ZSMC: {:<10.4f}, test log_ZSMC: {:<10.4f}".format(i+1, log_ZSMC_train, log_ZSMC_test))
                
                if loss_merged is None:
                    log_ZSMC_train_summary = tf.summary.scalar('log_ZSMC_train', log_ZSMC_train)
                    log_ZSMC_test_summary = tf.summary.scalar('log_ZSMC_test', log_ZSMC_test)
                    loss_merged = tf.summary.merge([true_log_ZSMC_summary, log_ZSMC_train_summary, log_ZSMC_test_summary])
                loss_summary = sess.run(loss_merged)
                writer.add_summary(loss_summary, i)

        A_val = A.eval()
        B_val = B.eval()
        Q_val = Q.eval()
        x_0_val = x_0.eval()


    sess.close()

    print("Done...")

    print("-------------------true val-------------------")
    print("A_true:\n", A_true)
    print("Q_true:\n", Q_true)
    print("B_true:\n", B_true)
    print("x_0_true:\n", x_0_true)
    print("-------------------optimized val-------------------")
    print("A_val:\n", A_val)
    print("Q_val:\n", Q_val)
    print("B_val:\n", B_val)
    print("x_0_val:\n", x_0_val)


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
 
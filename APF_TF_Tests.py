import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import pdb


from datetools import addDateTime
import sys
from optparse import OptionParser
import pickle
import os


cur_date = addDateTime()
parser = OptionParser()

parser.add_option("--rltdir", dest='rltdir', default='Experiment')
args = sys.argv
(options, args) = parser.parse_args(args)

local_rlt_root = './rslts/APF_TensorFlow/'
local_rlt_dir = local_rlt_root + options.rltdir + cur_date + '/'
RLT_DIR = local_rlt_dir

if not os.path.exists(RLT_DIR): os.makedirs(RLT_DIR)


print("Akwaaba!")
print(tf.__version__)


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

def SymbPLDS(A,Q,Qinv,B,ones,x_0,mu,x,y):
    """ Define the log likelihood symbolically for PLDS """
    # Change me to operate on higher rank tensors
    DeltaX = x - mu
    DeltaXQinv = tf.matmul(DeltaX,Qinv)
    DeltaXQinvDeltaX = tf.matmul(DeltaXQinv,DeltaX,transpose_b=True)
    expBX = tf.exp(tf.matmul(x,B,transpose_b=True))
    expBXOne = tf.matmul(expBX,ones,transpose_b=True)
    BXY = tf.matmul(y,tf.matmul(x,B,transpose_b=True),transpose_b=True)
    lnP = -0.5*DeltaXQinvDeltaX - expBXOne + BXY
    return lnP


def LaplaceApprox(A_DzxDz,  Q_DzxDz, Qinv_DzxDz, B_DyxDz, Zprev_Dz, y_Dy, ones_Dy, mu_Dz = None, niter=2, debug_mode=False):
    """ Computes Laplace Approx using an FPI for the first and second moments differentiating the log posterior """
    #ones_Dy = tf.Variable(np.expand_dims(np.ones(B_DyxDz.shape[0]),axis=0), dtype=tf.float32)
    Z_Dz = Zprev_Dz
    mu_Dz = tf.matmul(Zprev_Dz, A_DzxDz)
    #pdb.set_trace()
    # Iterate over FPIs for first and second moments:
    for i in range(niter):
        # Compute FPI for the mean
        BtOnes_Dz = tf.matmul(ones_Dy,B_DyxDz)
        #QBtOnes_Dz = tf.matmul(BtOnes_Dz, Q_DzxDz)
        BZ_Dy = tf.matmul(Z_Dz, B_DyxDz, transpose_b=True)
        expBZ_Dy = tf.exp(BZ_Dy)
        #expBZOnes_1x1 = tf.matmul(expBZ_Dy, ones_Dy, transpose_b=True)
        BtexpBZ_Dz = tf.matmul(expBZ_Dy, B_DyxDz)
        BtY_Dz = tf.matmul(y_Dy, B_DyxDz)
        QBtY_Dz = tf.matmul(BtY_Dz, Q_DzxDz)
        #Z_Dz = - QBtOnes_Dz * expBZOnes_1x1 + QBtY_Dz + mu_Dz
        Z_Dz = - np.dot(Q_DzxDz, BtexpBZ_Dz) + QBtY_Dz + mu_Dz
        if debug_mode:
            print("iter %i, mean:\n"%i, Z_Dz)
        # Compute FPI for the Hessian
        expBZ_DyxDy = tf.diag(tf.exp(tf.matmul(B_DyxDz,Z_Dz,transpose_b=True))[:,0])
        BtexpBZ_DzxDy = tf.matmul(B_DyxDz, expBZ_DyxDy, transpose_a=True)
        BtexpBZB_DzxDz = tf.matmul(BtexpBZ_DzxDy, B_DyxDz)
        H_DzxDz = BtexpBZB_DzxDz + Qinv_DzxDz
        if debug_mode:
            print("iter %i, Hessian:\n", H_DzxDz)
    # Compute the inverse normalization to approximate the integral
    SqInvDet = ((1./tf.matrix_determinant(H_DzxDz))**(1./2.))
    PiTerm = ((2*tf.constant(np.pi)))**(tf.shape(Qinv_DzxDz,out_type=tf.float32)[0]/2)
    #Pstar = make_mvn_pdf(mu_Dz, Q_DzxDz)(Z_Dz) * make_poisson(y_Dy)(np.dot(B_DyxDz, Z_Dz))
    Ztilde_1x1 = SqInvDet * PiTerm # * Pstar
    return Ztilde_1x1

def TensorLaplaceApprox(A_DzxDz,  Q_DzxDz, Qinv_DzxDz, B_DyxDz, Zprev_NxDz, y_Dy, ones_Dy, mu_Dz = None, niter=2, debug_mode=False):
    """ Broadcasts Laplace Approx computing N integrals. Using an FPI for the first and second moments differentiating the log posterior """
    #ones_Dy = tf.Variable(np.expand_dims(np.ones(B_DyxDz.shape[0]),axis=0), dtype=tf.float32)
    #pdb.set_trace()
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
        Z_NxDz = - QBtexpBZ_NxDz + QBtY_Dz + mu_NxDz
        if debug_mode:
            print("iter %i, mean:\n"%i, Z_NxDz)
        # Compute FPI for the Hessian
        #expBZ_DyxDy = tf.diag(tf.exp(tf.matmul(B_DyxDz,Z_NxDz,transpose_b=True))[:,0])
        expBZ_NxDyxDy = tf.matrix_diag(tf.exp((tf.matmul(Z_NxDz,B_DyxDz, transpose_b=True))))
        BtexpBZ_NxDzxDy = tf.einsum('ijk,jh->ihk', expBZ_NxDyxDy, B_DyxDz)
        #BtexpBZB_DzxDz = tf.matmul(BtexpBZ_NxDzxDy, B_DyxDz)
        BtexpBZB_NxDzxDz = tf.einsum('ijk,kh->ijh', BtexpBZ_NxDzxDy, B_DyxDz)
        H_NxDzxDz = BtexpBZB_NxDzxDz + Qinv_DzxDz
        if debug_mode:
            print("iter %i, Tensor of Hessians:\n", H_NxDzxDz)
    # Compute the inverse normalization to approximate the integral
    SqInvDet = ((1./tf.sqrt(tf.matrix_determinant(H_NxDzxDz)))**(1./2.))
    PiTerm = ((2*tf.constant(np.pi)))**(tf.shape(Qinv_DzxDz,out_type=tf.float32)[0]/2)
    Ztilde_Nx1 = SqInvDet * PiTerm
    return Ztilde_Nx1




def APF_SMC(X, W, k, n_particles, A, B, Q, x_0, obs, sess):

    """ Implements the Fully Adapted Auxiliary Particle Filter via TensorFlow """
    # Grab dimensions, define precision matrix
    Dy, Dx = B.shape
    time = obs.shape[0]
    Qinv = tf.matrix_inverse(Q)
    ones_Dy = np.expand_dims(np.ones(obs.shape[1], dtype=np.float32), axis=0)

    #pdb.set_trace()
    # Broadcast first iteration over particles
    # Define proposal distribution
    #q_uno = tfd.MultivariateNormalDiag(loc = np.zeros(Dx,dtype=np.float32), scale_diag = np.ones(Dx,dtype=np.float32))
    q_uno = tfd.MultivariateNormalDiag(loc=x_0[0], scale_diag=np.ones(Dx, dtype=np.float32))

    # Sample from the proposal distribution
    q_uno_samp = tf.cast(q_uno.sample(n_particles), dtype=tf.float32)

    # Assign latent variables sampled from the proposal to initialize particle values
    Xset = tf.assign(X[:,0,:], q_uno_samp)
    sess.run(Xset)

    # Compute probabilities from the proposal
    q_uno_probs = q_uno.prob(q_uno_samp)

    # Define initial transition density
    f_nu_uno = tfd.MultivariateNormalDiag(loc = x_0[0], scale_diag = np.ones(Dx,dtype=np.float32))

    # Sample from transition density and compute probabilities
    #f_nu_probs = f_nu_uno.prob(tf.cast(f_nu_uno.sample(n_particles), dtype=tf.float32))
    f_nu_probs = f_nu_uno.prob(q_uno_samp)

    # Define initial emission density
    g_uno = tfd.Poisson(log_rate=tf.matmul(B, X[:, 0, :], transpose_b=True))
    #pdb.set_trace()
    ### obs[0, :]
    # Sample from emission density and compute probabilities
    obs_uno = np.tile(obs[0,:], (n_particles,1)).astype(np.float32).T#tf.cast(obs[0, :], dtype=tf.float32)
    #g_uno_probs = tf.reduce_prod(g_uno.prob(g_uno.sample(1)), axis=(0,1))
    g_uno_probs = tf.reduce_prod(g_uno.prob(obs_uno), axis=(0))

    # Compute initial importance weights
    prob = g_uno_probs * f_nu_probs / q_uno_probs
    Wset = tf.assign(W[:,0], prob)
    sess.run(Wset)

    #if False:
    # Main loop of program, revisit and broadcast inner loop if possible
    for t in range(1,time):
        start_time = datetime.now()
        print("time: ", t)

        # Broadcast particle operations at each time point
        assign_k = tf.assign(k[:,t], TensorLaplaceApprox(A, Q, Qinv, B, X[:,t-1,:], np.expand_dims(obs[t,:].astype(np.float32),axis=0), ones_Dy, 2))
        #assign_k = tf.assign(k[:, t], tf.ones([n_particles]))
        sess.run(assign_k)
        #print("particle mass:\n", k[:,t].eval())
        assign_w = tf.assign(W[:, t - 1], W[:, t - 1] * k[:, t])
        sess.run(assign_w)
        # Resample - this is annoying. Absorb both operations within one loop?
        #pdb.set_trace()
        ancestors = tf.multinomial(tf.log(tf.expand_dims(W[:,t-1]/tf.reduce_sum(W[:,t-1],axis=0),axis=0)), n_particles)
        print("ancestors:\n", ancestors.eval())

        #ancestors = tf.expand_dims(W[:,t-1]/tf.reduce_sum(W[:,t-1],axis=0),axis=1)
        Xtilde = tf.gather(X[:,t-1,:], ancestors)[0]
        #Xtilde = ancestors


        # Assign resampled latent values to particles at the previous time point
        assign_prev_x = tf.assign(X[:,t-1,:], Xtilde)
        sess.run(assign_prev_x)

        #pdb.set_trace()
        # Broadcast second iteration over particles
        # Define proposal distribution
        q_t = tfd.MultivariateNormalDiag(loc=X[:,t-1], scale_diag=np.ones(Dx, dtype=np.float32))

        # Sample from the proposal distribution
        q_t_samp = tf.cast(q_t.sample(1), dtype=tf.float32)
        #pdb.set_trace()
        # Assign latent variables sampled from the proposal to initialize particle values
        assign_x = tf.assign(X[:,t], q_t_samp[0])
        sess.run(assign_x)

        # Compute probabilities from the proposal
        q_t_probs = q_t.prob(q_t_samp)

        # Define transition density
        f_t = tfd.MultivariateNormalDiag(loc=X[:, t - 1], scale_diag=np.ones(Dx, dtype=np.float32))

        # Sample from transition density and compute probabilities
        #f_t_probs = f_t.prob(tf.cast(f_t.sample(1), dtype=tf.float32))
        f_t_probs = f_t.prob(tf.cast(X[:,t], dtype=tf.float32))

        # Define emission density
        g_t = tfd.Poisson(log_rate=tf.matmul(B, X[:, t], transpose_b=True))

        # Sample from emission density and compute probabilities
        #g_t_probs = tf.reduce_prod(g_t.prob(g_t.sample(1)), axis=(0, 1))
        obs_t = np.tile(obs[t,:], (n_particles, 1)).astype(np.float32).T
        g_t_probs = tf.reduce_prod(g_t.prob(obs_t), axis=0)

        assign_w = tf.assign(W[:, t], g_t_probs * f_t_probs[0] / k[:, t] * q_t_probs[0])
        sess.run(assign_w)
        print("Resampled weights:\n", W[:, t].eval())
        end_time = datetime.now()
        print("Duration:", (end_time - start_time))

    return X, W, k


def computeZSMC(W,sess):
    """ Constructs the likelihood and log likelihood from the weights """
    #pdb.set_trace()

    avg = tf.reduce_mean(W[0:, :], axis=0)
    lnZ_SMC = tf.reduce_sum(tf.log(avg))

    return lnZ_SMC








if __name__ == '__main__':

    """ Test that shit is working """
    np.random.seed(1)
    # Define PLDS
    A = np.diag([0.95, 0.92])
    Q = np.asarray([[1., 0], [0, 1.]])
    Qinv = np.linalg.inv(Q)
    #B = np.random.rand(50,2)
    #B = np.array([[0.84237965, 0.55764607],
    #            [0.52567491, 0.88937252]])
    B = np.diag([.85,.92])
    T = 25
    x_0 = np.random.randn(1,2)#np.array([[2., 3.]])#np.random.randn(1,2)
    mu = np.dot(x_0,A)
    ones = np.expand_dims(np.ones(B.shape[0]),axis=0)
    # Create and plot synthetic data
    [hidden, obs] = makePLDS(A,B,Q,T,x_0)
    plt.plot(hidden[:,:], c='red')
    #plt.plot(obs[:,:], c='blue')
    plt.savefig(RLT_DIR + "Training Data")
    plt.show()

    # Test functions defined symbolically
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    #logP = SymbPLDS(A,Q,Qinv,B,ones,x_0,mu,hidden[0],obs[0])
    #print(logP.eval().shape)
    sess.close()

    A = tf.Variable(A, dtype=tf.float32, name='A')
    B = tf.Variable(B, dtype=tf.float32, name='B')
    Q = tf.Variable(Q, dtype=tf.float32, name='Q')
    Qinv = tf.Variable(Qinv, dtype=tf.float32, name='Qinv')
    mu = tf.Variable(mu, dtype=tf.float32, name='mu')
    ones = tf.Variable(ones, dtype=tf.float32, name='Ones')
    myX = tf.Variable(np.expand_dims(hidden[0], axis=0), dtype=tf.float32, name='MyX')
    myY = tf.Variable(np.expand_dims(obs[0], axis=0), dtype=tf.float32, name='MyY')
    x_0 = tf.Variable(x_0, dtype=tf.float32, name='x_0')
    lnP = SymbPLDS(A, Q, Qinv, B, ones, x_0, mu, myX, myY)


    # Test log probability of PLDS
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    print(lnP.eval())
    print("log P:", sess.run(lnP))
    myg = tf.gradients(lnP, [B, Qinv, myX])
    print("The gradients wrt B:\n", sess.run(myg[0]))
    print("The gradients wrt Qinv:\n", sess.run(myg[1]))
    print("The gradients wrt X:\n", sess.run(myg[2]))
    print([g for g in myg])
    sess.close()

    
    # Test Laplace Approx of the function below
    # p(y_n|x_n-1) = \int f(x_n|x_{n-1})g(y_n|x_n) dx_n
    Zpart = LaplaceApprox(A,Q,Qinv,B,x_0,myY,ones)
    ZZ = tf.Variable(hidden, dtype=tf.float32, name='ZZ')
    YY = tf.Variable(obs, dtype=tf.float32, name='YY')
    Zpart_Nx1 = TensorLaplaceApprox(A, Q, Qinv, B, ZZ, YY, ones)
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    sess.run(init)
    print(Zpart.eval())
    print("Zpart:", sess.run(Zpart))
    print("Broadcasted Laplace:", sess.run(Zpart_Nx1))
    sess.close()
    #assert False


    # Test basics of first loop in APF
    #with tf.Graph().as_default():

    #pdb.set_trace()

    # Define TensorFlow variables
    n_particles = 500
    time = T
    Dx = A.shape[0]
    X = tf.Variable(np.zeros((n_particles, time, Dx)), dtype=tf.float32, name='Z', trainable=False)
    W = tf.Variable(np.zeros((n_particles, time)), dtype=tf.float32, name='Weights', trainable=True)
    k = tf.Variable(np.zeros((n_particles, time)), dtype=tf.float32, name='marginals', trainable=False)
    init = tf.global_variables_initializer()


    ## Define learning rate
    lr = 0.01

    with tf.Session() as sess:
        sess.run(init)
        #writer = tf.summary.FileWriter("output", sess.graph)
        myAPF = APF_SMC(X, W, k, n_particles, A, B, Q, x_0, obs, sess)
        #pdb.set_trace()
        Particles = myAPF[0].eval()
        Weights = myAPF[1].eval()
        Partitions = myAPF[2].eval()
        print("particles:\n", Particles)
        print("weights:\n", Weights)
        print("partitions:\n", Partitions)
        #pdb.set_trace()
        ln_Z_SMC = computeZSMC(Weights, sess)
        myG = tf.gradients(ln_Z_SMC, [A])
        print("gradients:\n", myG)

        writer = tf.summary.FileWriter(RLT_DIR + './logs/1/train')
        #merge = tf.summary.merge_all()
        writer.add_graph(sess.graph)

        #train_op = tf.train.GradientDescentOptimizer(lr).minimize(-ln_Z_SMC)

    sess.close()

    plt.figure()
    plt.plot(Particles[:,:,0].T, alpha=0.01, c='black')
    plt.plot(hidden[:, 0], c='yellow')
    sns.despine()
    plt.savefig(RLT_DIR + "Filtered Paths Dim 1")
    plt.show()

    plt.figure()
    plt.plot(Particles[:,:,1].T,  alpha=0.01, c='black')
    plt.plot(hidden[:, 1], c='yellow')
    sns.despine()
    plt.savefig(RLT_DIR + "Filtered Paths Dim 2")
    plt.show()



    """
    init = tf.global_variables_initializer()
    sess = tf.InteractiveSession()
    #writer = tf.summary.FileWriter("output", sess.graph)
    sess.run(init)

    myAPF = APF_SMC(10, A, B, Q, x_0, obs)
    print("probabilities:\n", myAPF[0].eval())
    print("probabilities:\n", myAPF[1].eval())
    sess.run(myAPF[1])
    pdb.set_trace()
    #writer.close()
    sess.close()
    """

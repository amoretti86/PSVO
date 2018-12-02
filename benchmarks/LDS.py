import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pykalman import KalmanFilter
import pdb


def checkLDS(T, A, C, Q, S):
    """
    Quick sanity check to make sure code works...
    generate and fit 1d -> 1d system
    """
    obs = np.zeros(T)
    hid = np.zeros(T)
    hid[0] = 4

    for i in range(1,T):
        hid[i] = A*hid[i-1] + Q*np.random.randn()
        obs[i] = C*hid[i] + S*np.random.randn()
        
    plt.figure()
    plt.title("Simulated Data")
    plt.plot(hid, c='blue')
    plt.plot(obs, c='red')
    plt.legend(['hidden', 'observed'])
    plt.show()
    
    mykf = KalmanFilter(initial_state_mean=4, n_dim_state=1, n_dim_obs=1, em_vars=['transition_matrices', 'observation_matrices', 'transition_covariance', 'observation_covariance'])
    mykf.em(obs,n_iter=200)
    
    plt.figure()
    myZ, mySig = mykf.smooth(obs)
    plt.title("Estimated States vs Ground Truth")
    plt.plot(myZ, c='red')
    plt.plot(hid, c='blue')
    plt.legend(['smoothed','true'])
    plt.show()
    
    return mykf.transition_matrices, mykf.observation_matrices, mykf.transition_covariance, mykf.observation_covariance
    
# test it out...
exp1 = checkLDS(100,.92, .85, .8, .7)
exp2 = checkLDS(125,.99, .7, .1, .1)

print("parameters learned:\n A,C,Q,Sigma:\n", exp2)


def checkMLDS(T, A, C, Q, S):
    """
    Quick sanity check to test multivariate LDS
    Generate Dx latent dynamics and fit Dy observation
    """
    Dx = A.shape[0]
    Dy = C.shape[0]
    obs = np.zeros([T,Dy])
    hid = np.zeros([T,Dx])


    for i in range(1,T):
        hid[i] = np.dot(A,hid[i-1]) + np.dot(Q,np.random.randn(Dx))
        obs[i] = np.dot(C,hid[i]) + np.dot(S,np.random.randn(Dy))
        
    plt.figure()
    plt.title("Simulated Data")
    plt.plot(hid, c='blue')
    plt.plot(obs, c='red')
    plt.legend(['hidden', 'observed'])
    plt.show()
    
    mykf = KalmanFilter(initial_state_mean=np.zeros(Dx), n_dim_state=Dx, n_dim_obs=Dy, em_vars=['transition_matrices', 'observation_matrices', 'transition_covariance', 'observation_covariance'])
    mykf.em(obs,n_iter=100)
    
    plt.figure()
    myZ, mySig = mykf.smooth(obs)
    plt.title("Estimated States vs Ground Truth")
    plt.plot(myZ, c='red')
    plt.plot(hid, c='blue')
    plt.legend(['smoothed','true'])
    plt.show()
    
    return mykf.transition_matrices, mykf.observation_matrices, mykf.transition_covariance, mykf.observation_covariance
    
    
# Define 2d dynamics and 3d observation and try to recover the 2d dynamics from the 3d observation...
model3 = checkMLDS(125, np.array([[0.99, .02],[.02, 0.99]]), np.array([[3., 3.,], [2., 2.], [4., 3.]]), .1*np.eye(2), .1*np.eye(3))

# Here I define 2d dynamics along with a 1d observation and try to recover the 2d dynamics from the 1d observation...
model4 = checkMLDS(125, np.array([[0.99, .02],[.02, 0.99]]), np.array([[.7, .5]]), .1*np.eye(2), .1)

class LDS:
    """
    Train an LDS with the EM algorithm, find latent paths using Kalman Filter and Smoother
    Here I iterate over individual trials calling the EM algorithm once on each trial...
    """
    def __init__(self, X, Dz):
        self.X = X
        self.NTrials, self.NTbins, self.Dx = X.shape
        self.Dz = Dz
        self.kf = KalmanFilter(n_dim_state=Dz, n_dim_obs=self.Dx,  em_vars=['transition_matrices', 'observation_matrices', 'transition_covariance', 'observation_covariance'])
        self.Z = np.zeros([self.NTrials, self.NTbins, self.Dz])
        self.filtered_paths = np.zeros([self.NTrials, self.NTbins, self.Dz])
        self.filtered_covar = np.zeros([self.NTrials, self.NTbins, self.Dz, self.Dz])
        self.smoothed_covar = np.zeros([self.NTrials, self.NTbins, self.Dz, self.Dz])
        
    def train(self, epochs):
        print("EM Algorithm Training...")
        for i in range(epochs):
            print("- Epoch %i" %i)
            for n in range(self.NTrials):
                print("-- Trial %i" %n)
                self.kf.em(self.X[n],n_iter=1)

        self.A = self.kf.transition_matrices
        self.Q = self.kf.transition_covariance
        self.C = self.kf.observation_matrices
        self.Sigma = self.kf.observation_covariance

    def inference(self):
        for n in range(self.NTrials):
            print("-- Trial %i" % n)
            self.filtered_paths[n], self.filtered_covar[n] = self.kf.filter(self.X[n])
            self.Z[n], self.smoothed_covar[n] = self.kf.smooth(self.X[n])


# Load data from file
datadir = '/Users/antoniomoretti/Desktop/dhern-ts_wcommona-b4b1ad88b3aa/data/fitzhughnagumo/'

with open(datadir + "datadict", 'rb') as handle:
    data = pickle.load(handle, encoding='latin1')
    
X = data['Ytrain']
# Fit an LDS to the FHN data mapping 1d obs -> 2d latent state
FHNLDS = LDS(X, Dz=2)

# Specify number of epochs to run EM
FHNLDS.train(epochs=50)

# Print transition matrices
print("A:\n", FHNLDS.A)
print("Q:\n", FHNLDS.Q)
print("C:\n", FHNLDS.C)
print("Sigma:\n", FHNLDS.Sigma)

# Perform filtering and smoothing
FHNLDS.inference()

# Plot latent states
plt.title("Latent States")
plt.plot(FHNLDS.Z[0]);

# Try fitting a single trial
FLDS = KalmanFilter(n_dim_state=2, n_dim_obs=1, em_vars=['transition_matrices', 'observation_matrices', 'transition_covariance', 'observation_covariance'])
plt.plot(X[0])

# Kalman Smoother and EM algorithm just copy observation, dynamics matrix returned is I
FLDS.em(X[0],n_iter=50)
print(FLDS.transition_matrices)
print(FLDS.observation_matrices)
mu, cov = FLDS.smooth(X[22])
plt.plot(mu)


# Check Lorenz Data
datadir = '/Users/antoniomoretti/Desktop/dhern-ts_wcommona-b4b1ad88b3aa/data/lorenz_d10_gaussian01/'

with open(datadir + "lorenzdatadict", 'rb') as handle:
    lorenz = pickle.load(handle, encoding='latin1')

# EM Algorithm fails to converge on 3-Tensor of (trials, tbins, Dy)
LorenzLDS = LDS(lorenz['Ytrain'], Dz = 3)
LorenzLDS.train(epochs=100)

# Try fitting a single trial
n = 64
lorenz_1 = KalmanFilter(n_dim_state=3, n_dim_obs=10, em_vars=['transition_matrices', 'observation_matrices', 'transition_covariance', 'observation_covariance'])
lorenz_1.transition_matrices
lorenz_1.em(lorenz['Ytrain'][n],n_iter=50)
lorenzmu, lorenzcov = lorenz_1.smooth(lorenz['Ytrain'][n])

# Plot inferred trajectory
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(lorenzmu[:,0],lorenzmu[:,1],lorenzmu[:,2], c='red')
ax.scatter(lorenzmu[0,0],lorenzmu[0,1],lorenzmu[0,2], c='red')
plt.show()


# Plot ground truth latent state
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(lorenz['State'][n][:,0],lorenz['State'][n][:,1],lorenz['State'][n][:,2], c='red')
plt.show()

# Print dynamics matrix
print(lorenz_1.transition_matrices)

def kstep(model, data, Ydata, k):
    """
    Compute k-step MSE to evaluate model fit
    """
    Yprime_next = []
    Ydata_next = []
    
    kMSE_list = np.zeros(k)
    SE_list = np.zeros(k)
    
    for i in range(k):

        Xconv_nxt = np.dot(data, model.transition_matrices, )
        Yprime_nxt = np.dot(model.observation_matrices, Xconv_nxt.T)
        
        Yprime_next.append(Yprime_nxt) 
        Ydata_next.append(Ydata[i+1:,:])
        
        Xconv = Xconv_nxt
        DY_nxt = Ydata[i+1:,:] - Yprime_nxt[:,i+1:].T
        kMSE_list[i] = np.mean(DY_nxt**2)
        SE_list[i] = np.sum(DY_nxt**2)
        
        print('MSE %i: ' %i, np.mean(DY_nxt**2))
    
    return kMSE_list, SE_list, Yprime_next, Ydata_next
    
kmse, se, yp, yd = kstep(lorenz_1, lorenzmu, lorenz['Ytrain'][n], 1)

# Compute the mean square total to normalize the MSE and compute R^2
MSTlist = np.zeros(1)
for i in range(1):
    err = np.zeros([lorenz['Ytrain'][64].shape[0]-i-1,lorenz['Ytrain'][64].shape[1]])
    err = lorenz['Ytrain'][64][i+1:,:] - np.mean(lorenz['Ytrain'][64][i+1:,:])
    MSTlist[i] = np.mean(err**2)
    
1 - kmse/MSTlist

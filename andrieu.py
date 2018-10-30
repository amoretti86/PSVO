import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

ntrials = 100
time=201


def andrieu(theta1, theta2, time):
    X = np.zeros(time)
    Y = np.zeros(time)
    X[0] = np.random.randn()*np.sqrt(5)
    for i in range(1,time):
        X[i] = theta1 * X[i-1] + 25*X[i-1]/(1+X[i-1]**2) + 8*np.sqrt(10) + np.cos(1.2*time) + np.random.randn()
        Y[i] = theta2 * X[i]**2 + np.sqrt(10)*np.random.randn()

    return X, Y

X, Y = andrieu(theta1=0.5, theta2=0.05, time=201)


plt.figure(figsize=(10,10))
plt.plot(X[1:], c='red')
plt.plot(Y[1:], c='blue')
plt.show()


Xdata = np.zeros([ntrials,time])
Ydata = np.zeros([ntrials,time])

for i in range(ntrials):
    Xdata[i], Ydata[i] = andrieu(.5, 0.05, time)


print("Plotting data...")
plt.figure(figsize=(12, 12))
plt.title("Filtered Time Series")
batch_size = 10
for i in range(ntrials):
    print("Figure %i" %i)
    plt.subplot(ntrials / batch_size, batch_size, i + 1)
    plt.plot(Xdata[i], c='red')
    plt.plot(Ydata[i], c='blue')
    sns.despine()
    plt.tight_layout()
plt.savefig("Training Data")
plt.show()
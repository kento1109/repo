import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from matplotlib import mlab
from sklearn.preprocessing import StandardScaler
import math

X = np.array([[1.,1.],[23.,45.],[2.,7.],[79.,80.],[30.,20.],[83.,71.]])
N = X.shape[0]
R = X.shape[1]
K = 3
maxItr = 20
limit = 0.1
np.random.seed(seed=32)

def init():

    global X
    #u = np.random.permutation(X)[:K]
    #u = np.array([X.min(axis=0),X.max(axis=0)])
    u = np.array([X.min(axis=0),X.mean(axis=0),X.max(axis=0)])
    cov = np.zeros([K*R,R])
    for k in range(K):
        cov[k*R:k*R+2,:] = np.dot(((X-u[k])).T,(X-u[k]))/N
    weight = np.array([0.3,0.3,0.4])
    #X = standarization(X)
    return u,cov,weight

def standarization(X):
    scaler = StandardScaler()
    scaler.fit(X)
    x_scaled = scaler.transform(X)
    return x_scaled

def estimate(u_,cov_,weight_):

    i = 0
    h = np.zeros([N,K])
    while i < N:
        p_all = 0
        j = 0
        p_j = np.zeros(K)
        while j < K:
            rv = multivariate_normal(u_[j], cov_[j*2:j*2+2])
            p_j[j] = rv.pdf(X[i])*weight_[j]
            p_all += p_j[j]
            j += 1
        h[i] = [p_/p_all for p_ in p_j]
        i += 1
    return h

def maximize(h_):

    j = 0
    while j < K:
        i = 0
        hx = 0
        h = np.array(h_[:,j], ndmin=2)
        u[j] = np.sum(h.T*X,axis=0)/h.sum()
        cov[j*2:j*2+2] = np.dot(((X-u[j])*h.T).T,(X-u[j]))/h.sum()
        weight[j] = h.sum()/N
        j += 1
    return u,cov,weight

def calc_likelihood(u_,cov_,weight_):

    i = 0
    while i < N:
        likelihood = 0
        j = 0
        p_j = np.zeros(K)
        while j < K:
            rv = multivariate_normal(u_[j], cov_[j*2:j*2+2])
            p_j[j] = rv.pdf(X[i])*weight_[j]
            likelihood += p_j[j]
            j += 1
        i += 1
    return math.log(likelihood)

if __name__ == '__main__':

    u,cov,weight = init() # initialization
    old_likelihood = 0
    for _itr in range(maxItr):
        h = estimate(u,cov,weight) # E-Step
        u,cov,weight = maximize(h) # M-Step
        likelihood = calc_likelihood(u,cov,weight)
        if abs(likelihood - old_likelihood) < limit:
            break
        old_likelihood = likelihood
    # plot
    xlist = np.linspace(0, 100, 100)
    ylist = np.linspace(0, 100, 100)
    x, y = np.meshgrid(xlist, ylist)
    for k in range(K):
        z =  mlab.bivariate_normal(x, y, np.sqrt(cov[k*2,0]), np.sqrt(cov[k*2+1,1]), u[k,0], u[k,1], cov[k*2,1])
        cs = plt.contour(x, y, z, 3, colors='k', linewidths=1)
    plt.plot(X[:,0:1],X[:,1:2],"bx")
    plt.plot(u[:,0:1],u[:,1:2],"ro")
    plt.show()
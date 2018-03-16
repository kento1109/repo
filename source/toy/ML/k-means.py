import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1.,1.],[3.,5.],[2.,2.],[5.,6.],[9.,9.],[8.,7.]])
K = 3
maxItr = 10
centroid = np.zeros([K,X.shape[1]])
distance = np.zeros([K])

np.random.seed(seed=32)
np.random.rand(3)

def init():
    C = np.random.randint(0,K,len(X))
    return C

def calc_centroid(cls):
    for c_num in range(K):
        centroid[c_num] =  np.sum(X[C==c_num], axis=0)/len(X[C==c_num])
    return centroid

def reassign_data(centroid):
    for (i,x) in enumerate(X):
        for c_num in range(K):
            distance[c_num] = np.linalg.norm((x - centroid[c_num]), ord=1) # L1 norm
        C[i] = distance.argmin()
    return C

if __name__ == '__main__':
    C = init() # initialization
    for _itr in range(maxItr):
        centroid = calc_centroid(C) # caclulate centroid
        C = reassign_data(centroid) # reassgin data
    plt.scatter(X[:,0],X[:,1],c=C)
    plt.scatter(centroid[:,0],centroid[:,1],c="b")
    plt.show()

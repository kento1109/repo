import numpy as np
from scipy.stats import gamma, poisson, multinomial, dirichlet
from scipy.misc import logsumexp
import random
import matplotlib.pyplot as plt


N = 500
K = 2
# make sample data
lambda_X = [5, 15]
X = []
fig = plt.figure()
edges = range(0, 30)
for i in range(K):
    ax = fig.add_subplot(2, 1, i + 1)
    samples = poisson.rvs(lambda_X[i], size=N / 2, random_state=1234)
    n, bins, patches = ax.hist(samples, bins=edges)
    X.extend(samples)
    plt.xlim(0, 30)
    plt.ylim(0, 20)
# data shuffle
random.shuffle(X)

MAXIter = 500
# hyper parameter
a = np.ones(K)
b = np.ones(K)
alpha = np.ones(K)
# set initial value
Lambda = gamma.rvs(a=a, scale=np.reciprocal(b), size=K)
Pi = dirichlet.rvs(alpha=alpha, size=1)
S = np.zeros((MAXIter, N, K))
# S = np.zeros((N, K))

for iter in range(MAXIter):
    for i, x in enumerate(X):
        # inference s
        tmp_eta = x * np.log(Lambda) - Lambda + np.log(Pi)
        z = logsumexp(tmp_eta)
        eta = np.exp(tmp_eta - z)
        s = multinomial.rvs(n=1, p=eta[0], size=1)
        S[iter, i, :] = s
    # inference lambda
    a = [np.dot(S[iter, :, k], X) + a[k] for k in range(K)]
    b = [np.sum(S[iter, :, k]) + b[k] for k in range(K)]
    Lambda = np.array([gamma.rvs(a=a[k], scale=np.reciprocal(b[k]), size=1) for k in range(K)]).reshape(1, 2)
    # inference pi
    alpha = [np.sum(S[iter, :, k]) + alpha[k] for k in range(K)]
    # print alpha
    Pi = dirichlet.rvs(alpha=alpha, size=1)
print Pi
plt.show()
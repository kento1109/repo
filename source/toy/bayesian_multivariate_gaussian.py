import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
z = np.dstack((x, y))

# parameter of prior gaussian distribution
hp_ms = [[-1.0, -1.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]]
hp_Sigmas = [[[2.0, 1.0], [1.0, 2.0]], [[0.2, 0.0], [0.0, 0.5]], [[3.0, 0.0], [0.0, 2.0]], [[2.0, -0.5], [-1.0, 2.0]]]

# parameter of posterior gaussian distribution
Lambda = np.array([[2.0, 0.0], [1.0, 2.0]])
mu = np.array([1.0, 1.0])

for i, (hp_m, hp_Sigma) in enumerate(zip(hp_ms, hp_Sigmas)):
    hp_m = np.array(hp_m)
    hp_Lambda = np.linalg.inv(np.array(hp_Sigma))
    plt.subplot(4, 4, i + 1)
    # m's prior gaussian distribution
    m = multivariate_normal.pdf(z, hp_m, hp_Lambda)
    plt.title('N=0', size=10)
    plt.contour(x, y, m, 3)
    plt.tick_params(labelleft='off')
    plt.tick_params(labelbottom='off')
    for j, N in enumerate([3, 10, 50]):
        plt.subplot(4, 4, 4 * (j + 1) + i + 1)
        # observed data
        X = multivariate_normal.rvs(mu, Lambda, N, random_state=1234)
        # plt.plot(X[:, 0], X[:, 1], "o", markersize=2, label='N=' + str(N))
        # update parameter
        new_Lambda = N * np.array(Lambda) + hp_Lambda
        sum_X = np.sum(X, axis=0)
        new_m = np.dot(np.linalg.inv(new_Lambda), (np.dot(Lambda, sum_X) + np.dot(hp_Lambda, hp_m)))
        # posterior distribution
        Sigma = np.linalg.inv(new_Lambda)
        pdf = multivariate_normal.pdf(z, new_m, Sigma)
        plt.title('N=' + str(N), size=10)
        plt.contour(x, y, pdf, 3)
        # plt.legend(loc="upper left")
        plt.tick_params(labelleft='off')
        plt.tick_params(labelbottom='off')
plt.show()
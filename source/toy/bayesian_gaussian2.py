# -*- coding: utf-8 -*-
""" １次元ガウス分布の学習（精度が未知の場合） """
import numpy as np
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

x_ = np.linspace(-8, 8, 100)
# parameter of prior gamma distribution
params = [[1.0, 1.0], [2.0, 0.5], [2.0, 1.0], [3.0, 0.5]]

mu_ = 0  # parameter of gaussian distribution

for i, param in enumerate(params):
    plt.subplot(4, 4, i + 1)
    a_, b_ = param[0], param[1]
    p_lambda_ = gamma.pdf(x=x_, a=a_, scale=np.reciprocal(b_))  # prior distribution
    sigma_ = np.sqrt(np.reciprocal(p_lambda_))
    plt.plot(x_, sigma_, '-')
    plt.title('a=' + str(a_) + ' b=' + str(b_) , size=10)
    plt.tick_params(labelleft='off')
    for j, N in enumerate([3, 10, 50]):
        plt.subplot(4, 4, 4 * (j + 1) + i + 1)
        x = norm.rvs(loc=1, scale=1, size=N)  # observed data
        sum_x = np.sum(x)
        a = (N / 2) + a_  # update a
        b = 0.5 * np.sum(x - mu_) ** 2 + b_  # update b
        lambda_ = gamma.pdf(x=x_, a=a, scale=np.reciprocal(b))  # parameter lambda distribution
        sigma_ = np.sqrt(np.reciprocal(lambda_))  # calculate scale
        X = norm.pdf(x=x_, loc=mu_, scale=sigma_)  # posterior distribution
        plt.plot(x_, X, '-', lw=1, label='N=' + str(N))
        plt.legend(loc="upper right")
        plt.tick_params(labelleft='off')
plt.show()

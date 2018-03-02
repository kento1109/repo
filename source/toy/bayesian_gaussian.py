# -*- coding: utf-8 -*-
""" １次元ガウス分布の学習（平均が未知の場合） """
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

x_ = np.linspace(-8, 8, 500)
# parameter of prior gaussian distribution
params = [[-1.0, 0.1], [1.0, 1.0], [-1.0, 0.1], [1.0, 1.0]]

lambda_ = 0.25  # parameter of gaussian distribution

for i, param in enumerate(params):
    plt.subplot(4, 4, i + 1)
    m, p_lambda_ = param[0], param[1]
    sigma_ = np.sqrt(np.reciprocal(p_lambda_))
    mu = norm.pdf(x=x_, loc=m, scale=sigma_)
    plt.plot(x_, mu, '-', lw=1, label='mu=' + str(m) + ' scale=' + str(sigma_))
    plt.title('m=' + str(m) + ' p_lambda=' + str(p_lambda_), size=10)
    plt.tick_params(labelleft='off')
    for j, N in enumerate([3, 10, 50]):
        plt.subplot(4, 4, 4 * (j + 1) + i + 1)
        x = norm.rvs(loc=0, scale=1, size=N)  # observed data
        sum_x = np.sum(x)
        lambda_ = N * p_lambda_ + lambda_  # update lambda
        m_ = (lambda_ + sum_x + p_lambda_ * m) / lambda_  # update m
        sigma_ = np.sqrt(np.reciprocal(lambda_))  # calculate scale
        X = norm.pdf(x=x_, loc=m_, scale=sigma_)
        plt.plot(x_, X, '-', lw=1, label='N=' + str(N))
        plt.legend(loc="upper right")
        plt.tick_params(labelleft='off')
plt.show()

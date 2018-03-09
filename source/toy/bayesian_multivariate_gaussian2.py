# -*- coding: utf-8 -*-
""" 多次元ガウス分布の学習（精度が未知の場合） """

from scipy.stats import wishart, multivariate_normal
import numpy as np

x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
z = np.dstack((x, y))

# parameter of prior wishart distribution
hp_Ws = [[[0.5, 1.0], [0.5, 1.5]], [[5.0, 0.0], [0.0, 2.0]], [[0.3, 0.0], [0.0, 0.5]], [[0.5, -0.5], [-1.0, 1.5]]]
hp_vs = [2, 3, 4, 5]

# parameter of posterior gaussian distribution
Lambda = np.array([[1.0, 0.5], [0.5, 2.0]])
mu = np.array([1.0, 1.0])

for i, (hp_v, hp_W) in enumerate(zip(hp_vs, hp_Ws)):
    for j, N in enumerate([3, 10, 50]):
        # observed data
        X = multivariate_normal.rvs(mu, Lambda, N, random_state=1234)
        # update parameter
        sum_xu = np.matrix(np.sum(X - mu, axis=0)).reshape(2, 1)
        new_W_inv = (sum_xu * sum_xu.T) + np.linalg.inv(hp_W)
        new_v = N + hp_v
        # posterior distribution
        new_Lambda = wishart.rvs(df=new_v, scale=np.linalg.inv(new_W_inv), size=N)
# -*- coding: utf-8 -*-
""" 多次元ガウス分布の学習（平均・精度が未知の場合） """

from scipy.stats import wishart, multivariate_normal
import numpy as np

x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
z = np.dstack((x, y))

# hyper parameter of prior distribution
hp_m = np.array([0.0, 0.0])
hp_beta = 2.0
hp_W = np.array([[0.5, 1.0], [0.5, 1.5]])
hp_v = 2.0
# parameter of posterior gaussian distribution
Sigma = np.array([[1.0, 0.5], [0.5, 2.0]])
mu = np.array([1.0, 1.0])

for j, N in enumerate([3, 10, 50]):
    # observed data
    X = multivariate_normal.rvs(mu, Sigma, N, random_state=1234)
    # update parameter
    sum_x = np.matrix(np.sum(X, axis=0))
    new_beta = N + hp_beta
    new_m = (1 / new_beta) * (sum_x + hp_beta * hp_m)
    hp_m_ = np.asmatrix(hp_m).reshape(2, 1)
    new_m_ = np.asmatrix(new_m).reshape(2, 1)
    X_ = np.matrix(X).reshape(2, N)
    new_W_inv = (X_ * X_.T) + (hp_beta * hp_m_ * hp_m_.T) - (new_beta * new_m_ * new_m_.T) + np.linalg.inv(hp_W)
    new_v = N + hp_v
    # posterior distribution
    new_Lambda = wishart.rvs(df=new_v, scale=np.linalg.inv(new_W_inv), size=N)
    new_mu = multivariate_normal.pdf(z, np.squeeze(np.asarray(new_m)), np.linalg.inv(new_beta * new_Lambda))

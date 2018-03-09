# -*- coding: utf-8 -*-
""" １次元ガウス分布の学習（平均・精度が未知の場合） """
import numpy as np
from scipy.stats import norm, gamma
import matplotlib.pyplot as plt

x_ = np.linspace(-8, 8, 100)
# parameter of prior distribution
ps_m = [-1, 0, 1, 2]
ps_beta = [1, 2, 3, 4]
ps_ab = [[1.0, 1.0], [2.0, 0.5], [2.0, 1.0], [3.0, 0.5]]


def calc_a(p_N=0, p_a=0.0):
    return (p_N / 2) + p_a


def calc_b(x_=np.array([0]), beta_hat=0.0, m_hat=0.0, p_beta=0.0, p_m=0.0, p_b=0.0):
    return 0.5 * (np.sum(x_ ** 2) + (p_beta * p_m ** 2) - (beta_hat * m_hat ** 2)) + p_b


def calc_beta(p_N=0, p_beta=0.0):
    return p_N + p_beta


def calc_m(x_=np.array([0]), beta_hat=0.0, p_beta=0.0, p_m=0.0):
    return 1 / beta_hat * (np.sum(x_) + p_beta * p_m)


for i, (m_, beta_, ab_) in enumerate(zip(ps_m, ps_beta, ps_ab)):
    plt.subplot(4, 4, i + 1)
    a_, b_ = ab_[0], ab_[1]
    # update parameter without observed data
    beta_hat = calc_beta(p_beta=beta_)
    m_hat = calc_m(beta_hat=beta_hat, p_beta=beta_, p_m=m_)
    a_hat = calc_a(p_a=a_)
    b_hat = calc_b(beta_hat=beta_hat, p_beta=beta_, p_m=m_, p_b=b_)
    # parameter lambda distribution
    p_lambda_ = gamma.pdf(x=x_, a=a_, scale=np.reciprocal(b_))
    sigma_ = np.sqrt(np.reciprocal(p_lambda_))
    # parameter mu distribution
    mu_ = norm.pdf(x=x_, loc=m_hat, scale=sigma_)
    plt.plot(x_, mu_, '-')
    plt.title('a=' + str(a_) + ' b=' + str(b_), size=10)
    plt.tick_params(labelleft='off')
    for j, N in enumerate([3, 10, 50]):
        plt.subplot(4, 4, 4 * (j + 1) + i + 1)
        x = norm.rvs(loc=1, scale=1, size=N)  # observed data
        # update parameter with observed data
        beta_hat = calc_beta(p_N=N, p_beta=beta_)
        m_hat = calc_m(x_=x, beta_hat=beta_hat, p_beta=beta_, p_m=m_)
        a_hat = calc_a(p_N=N, p_a=a_)
        b_hat = calc_b(x_=x, beta_hat=beta_hat, m_hat=m_hat, p_beta=beta_, p_m=m_, p_b=b_)
        # parameter lambda distribution
        lambda_ = gamma.pdf(x=x_, a=a_hat, scale=np.reciprocal(b_hat))
        sigma_ = np.sqrt(np.reciprocal(lambda_))  # calculate scale
        X = norm.pdf(x=x_, loc=m_hat, scale=sigma_)  # posterior distribution
        plt.plot(x_, X, '-', lw=1, label='N=' + str(N))
        plt.legend(loc="upper right")
        plt.tick_params(labelleft='off')
plt.show()

# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import beta
from scipy.stats import bernoulli
from matplotlib import pyplot as plt

# bernoulli distribution parameter
p = 0.3

# hyper parameter
params = [[1, 1], [3, 3], [15, 5], [3, 7]]

for i, param in enumerate(params):
    plt.subplot(4, 4, i + 1)
    alpha_, beta_ = param[0], param[1]
    rv = beta(alpha_, beta_)
    s = np.linspace(0, 1, 1000)
    y = rv.pdf(s)
    plt.plot(s, y, '-', lw=1, label='N=0')
    plt.title('alpha=' + str(alpha_) + ' beta=' + str(beta_), size=10)
    plt.tick_params(labelleft='off')
    plt.legend(loc="lower right")
    for j, N in enumerate([5, 20, 100]):
        plt.subplot(4, 4, 4 * (j + 1) + i + 1)
        X = bernoulli.rvs(p, size=N)  # 観測データの作成
        xn = len(np.where(X == 1)[0])  # x=1の数
        alpha_ = xn + alpha_
        beta_ = N - xn + beta_
        rv = beta(alpha_, beta_)
        s = np.linspace(0, 1, 1000)
        y = rv.pdf(s)
        plt.plot(s, y, '-', lw=1, label='N=' + str(N))
        plt.tick_params(labelleft='off')
        plt.legend(loc="lower right")
plt.show()

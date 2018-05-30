import numpy as np

from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000,
                           n_classes=2,
                           weights=[0.99, 0.01],
                           random_state=1)

# print len(np.where(y==0)[0])
# print len(np.where(y==1)[0])

from sklearn.linear_model import LogisticRegression

# clf = LogisticRegression()
clf = LogisticRegression(class_weight='balanced')
clf.fit(X, y)
y_pred = clf.predict(X)

from sklearn.metrics import confusion_matrix
print(clf.score(X, y))
print(confusion_matrix(y, y_pred))

z = np.dot(X, clf.coef_.T) + clf.intercept_
logit = lambda z: 1 / (1 + np.exp(-z))

import matplotlib.pyplot as plt

linewidths = 0.1

plt.scatter(z, logit(z), c=y_pred, cmap='bwr', linewidths=linewidths)
plt.scatter(z, y, c="black", marker='^', linewidths=linewidths)
plt.show()

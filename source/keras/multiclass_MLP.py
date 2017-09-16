# L2正則化、ドロップアウトもやってみた。
import numpy as np
from sklearn.datasets import load_iris
from sklearn import model_selection
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras import regularizers
from keras.layers import Dropout
iris = load_iris()
data = iris['data']
target = iris['target']
feats = iris.data.shape[1]
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)
# one-hot-encoding
target = np_utils.to_categorical(target)
# separate data
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data, target, test_size=0.2, random_state=0)
model = Sequential()
model.add(Dense(32,input_dim=feats))
model.add(Dropout(0.5))
model.add(Dense(64, input_dim=64, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=300)
loss_and_metrics = model.evaluate(X_test,y_test,batch_size=32, verbose=0)
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))

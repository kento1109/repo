import numpy as np
from sklearn.datasets import load_iris
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.optimizers import SGD
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
model.add(Dense(3,input_dim=feats))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=300, batch_size=32, verbose=1)

loss_and_metrics = model.evaluate(X_test,y_test,batch_size=32, verbose=0)
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))

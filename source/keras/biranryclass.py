import numpy as np
from sklearn.datasets import load_iris
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
iris = load_iris()
data = iris['data']
target = iris['target']
feats = iris.data.shape[1]
# separate data
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data[target != 2], target[target != 2], test_size=0.2, random_state=0)
model = Sequential()
model.add(Dense(1,input_shape=(feats,)))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.01)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1)

loss_and_metrics = model.evaluate(X_test,y_test,batch_size=32, verbose=0)
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))
loss:0.0893804505467 accuracy:1.0

predict = model.predict_classes(X_test)
print "true:",y_test
print "pred:",[item for sublist in predict for item in sublist]

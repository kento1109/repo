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

true: [0 1 0 1 1 1 0 1 1 1 1 1 1 0 0 0 0 0 0 0]
pred: [0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]

"""
精度は100％！

predict: [[ 0.12239181]
 [ 0.94982004]
 [ 0.08861753]
 [ 0.94957912]
 [ 0.92901975]
 [ 0.8497709 ]
 [ 0.05486129]
 [ 0.95098931]
 [ 0.95795125]
 [ 0.90756142]
 [ 0.9437784 ]
 [ 0.91479629]
 [ 0.95380193]
 [ 0.08093811]
 [ 0.08852565]
 [ 0.12906139]
 [ 0.04684119]
 [ 0.14008878]
 [ 0.03719734]
 [ 0.13161948]]
＊二値の場合、クラス１に属する確率
"""

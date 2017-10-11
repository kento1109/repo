# -*- coding: utf-8-*-
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import InputLayer
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.models import model_from_json
from sklearn import model_selection

data_num = 2000
np.random.seed(14)


def make_data():
    i = 0
    xdata = []
    ydata = []
    while i < data_num:
        length = np.random.randint(1, 5)
        data = np.random.randint(0, 5, length)
        xdata.append(list(data.reshape(length, 1)))
        ydata.append(sum(data))
        i += 1

    return xdata, ydata


xdata, ydata = make_data()

maxlen = max([len(x) for x in xdata])

xdata = sequence.pad_sequences(xdata, maxlen=maxlen)
ydata = np.array(ydata).reshape(data_num, 1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    xdata, ydata, test_size=0.1, random_state=0)

in_out_dims = 1
hidden_dims = 8

# model = Sequential()
# model.add(InputLayer(batch_input_shape=(None, maxlen, in_out_dims)))
# model.add(
#     SimpleRNN(units=hidden_dims, use_bias=False, batch_input_shape=(None, maxlen, in_out_dims), return_sequences=False))
# model.add(Dense(in_out_dims))
# model.add(Activation("linear"))
# model.compile(loss="mean_squared_error", optimizer="rmsprop")
# model.fit(X_train, y_train, batch_size=100, epochs=50, validation_split=0.1)
#
# model_json_str = model.to_json()
# open('simple_rnn_model.json', 'w').write(model_json_str)
# model.save_weights('simple_rnn_weights.h5')

# モデルを読み込む
model = model_from_json(open('simple_rnn_model.json').read())

# 学習結果を読み込む
model.load_weights('simple_rnn_weights.h5')

print np.round(model.predict(X_test))[0:11]
print y_test[0:11]

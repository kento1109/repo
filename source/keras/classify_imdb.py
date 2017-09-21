from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Flatten, Input
from keras.layers.merge import Concatenate
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb

# set parameters:
word_index = 5000
maxlen = 500
batch_size = 32
embedding_dims = 32
filters = 32
kernel_list = [2,3,4]
hidden_dims = 250
epochs = 2

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=word_index)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
tokens_input = Input(name='input', shape=(maxlen,))

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
emd = Embedding(word_index,
                embedding_dims,
                input_length=maxlen)(tokens_input)

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
convolutions = []
for kernel_size in kernel_list:
    conv = Conv1D(filters,
                  kernel_size,
                  padding='same',
                  activation='relu')(emd)
    maxp = MaxPooling1D(pool_size=2)(conv)
    flt = Flatten()(maxp)
    convolutions.append(flt)
flt = Concatenate()(convolutions)

# We add a vanilla hidden layer:
hidden = Dense(hidden_dims, activation='relu')(flt)
# We project onto a single unit output layer, and squash it with a sigmoid:
output = Dense(1, activation='sigmoid')(hidden)
model = Model(input=tokens_input, output=output)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

loss_and_metrics = model.evaluate(x_test,y_test,batch_size=batch_size, verbose=1)
print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))

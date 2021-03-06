import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.merge import Concatenate
from keras.layers import Merge
from keras.layers import Input
from keras.utils.vis_utils import plot_model
from keras.models import Model

word_index = 1000 # # the number of vocabularies
embedding_dims = 300 # embedding dimmensions
maxlen = 100  # input dimensions
filters = 16
kernel_size = 3
kernel_list = [2,3,4]
hidden_dims = 64
batch_size = 32

tokens_input = Input(name='input', shape=(maxlen,))
emd = Embedding(word_index+1,
              embedding_dims,
              input_length=maxlen)(tokens_input)

conv = Conv1D(filters,
              kernel_size,
              activation='relu',
              padding='same')(emd)
              
maxp = MaxPooling1D(pool_size=2)(conv)

flt = Flatten()(maxp)

hidden = Dense(hidden_dims, activation='relu')(flt)

output = Dense(1, activation='sigmoid')(hidden)

model = Model(input=tokens_input, output=output)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

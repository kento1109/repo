# -*- coding: utf-8-*
import re
import MeCab
import cPickle
from keras.preprocessing import text
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers.core import Flatten
from keras.callbacks import CSVLogger


def main():
    # set parameters:
    batch_size = 16
    embedding_dims = 128
    filters = 64
    kernel_size = 3
    hidden_dims = 250
    epochs = 10
    num_classes = 5

    # ---- 前処理（ここから）----------------------------------------- #
    label, sentence = cPickle.load(open("CharcterFeature.p", "rb"))

    print "a number of sentence", len(sentence)
    maxlen = max([len(sent) for sent in sentence])
    print "max length of sentences", maxlen

    # 単語をtoken化
    tokenizer = text.Tokenizer()
    tokenizer.fit_on_texts(sentence)
    # マッピング辞書の作成
    word_index = tokenizer.word_index

    print "a number of vocabulary", len(word_index)

    cPickle.dump(word_index, open("word_index.p", "wb"))

    # 文書をidに変換
    sequences = tokenizer.texts_to_sequences(sentence)
    data = pad_sequences(sequences, maxlen=maxlen, padding='post')

    # ラベルをカテゴリデータに変換
    label = to_categorical(label)

    # print str(sorted(word_index.items(), key=lambda x: x[1])).decode('string-escape')

    print('x_train shape:', data.shape)

    print('Build model...')
    model = Sequential()

    model.add(Embedding(input_dim=len(word_index) + 1,
                        output_dim=embedding_dims,
                        input_length=maxlen))
    # model.add(Embedding(input_dim=len(word_index),
    #                     output_dim=embedding_dims,
    #                     input_length=maxlen))

    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=1))

    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    model.add(Dense(hidden_dims))
    model.add(Activation('relu'))

    model.add(Dropout(0.5))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print model.summary()

    callbacks = [CSVLogger("CNN_history.csv")]

    # model.fit(data, label,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           callbacks=callbacks)

    model.fit(data, label,
              batch_size=batch_size,
              epochs=epochs,
              callbacks=callbacks)

    model_json_str = model.to_json()
    open('CNN_model.json', 'w').write(model_json_str)
    model.save_weights('CNN_weights.h5')


if __name__ == '__main__':
    main()

#coding: utf-8
import base
from sklearn import model_selection
from gensim import corpora
from gensim import models
from gensim import matutils
from gensim.models import KeyedVectors
import numpy as np
import json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Flatten
from keras.utils.vis_utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text

# set parameters:
max_features = 5000
maxlen = 500
batch_size = 32
embedding_dims = 300
filters = 32
kernel_size = 3
hidden_dims = 250
epochs = 4

def Preprocessing(io,nlp,polarity):

    io = base.base_io(base_dir='./txt_sentoken/')
    dataset = io.Files2dataset(target_dir=polarity)

    # dataset = dataset[0:100]

    dataset = nlp.tokenize(dataset=dataset)
    dataset = nlp.stemming(dataset=dataset)
    dataset = [" ".join(sentence) for sentence in dataset]
    # labeling
    label = [1 if polarity == v_pos else 0 for i in range(len(dataset))]

    return dataset,label

def load_vocab(vocab_path='./Word2Vec.vocab'):
    """
    Load word -> index and index -> word mappings
    :param vocab_path: where the word-index map is saved
    :return: word2idx, idx2word
    """

    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word

def word2vec_embedding_layer(embeddings_path='embeddings.npz'):
    """
    Generate an embedding layer word2vec embeddings
    :param embeddings_path: where the embeddings are saved (as a numpy file)
    :return: the generated embedding layer
    """

    weights = np.load(open(embeddings_path, 'rb'))
    layer = Embedding(input_dim=weights.shape[0],
                      output_dim=weights.shape[1],
                      weights=[weights])
    return layer

if __name__ == '__main__':

    v_pos = 'pos'
    v_neg = 'neg'

    io = base.base_io(base_dir='./txt_sentoken')

    # Preprocessing
    nlp = base.base_nlp()
    pos_dataset, pos_label = Preprocessing(io,nlp,polarity=v_pos)
    neg_dataset, neg_label = Preprocessing(io,nlp,polarity=v_neg)
    dataset_all = pos_dataset + neg_dataset
    label_all = pos_label + neg_label

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(dataset_all)
    sequences = tokenizer.texts_to_sequences(dataset_all)
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=maxlen)

    weights = np.load(open('Word2Vec.weights', 'rb'))
    word2idx, idx2word = load_vocab()

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dims))
    print "**"
    for word, idx in word_index.items():
        if word in word2idx: # words not found in embedding index will be all-zeros.
            key = word2idx.get(word)
            print word
            embedding_matrix[key] = weights[key]
    print "**"
    # np.savetxt('embedding_matrix.csv', embedding_matrix, delimiter=',')

    x_train, x_test, y_train, y_test = model_selection.train_test_split(data, label_all,
                                                        test_size=0.2, random_state=0)

    print('Pad sequences (samples x time)')

    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(len(word_index) + 1,
                        embedding_dims,
                        weights=[embedding_matrix],
                        input_length=maxlen))
    # model.add(Embedding(max_features,
    #                     embedding_dims,
    #                     # weights=[embedding_matrix],
    #                     input_length=maxlen))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='same',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.summary()
    plot_model(model, to_file='model2.png', show_shapes=True)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    # loss_and_metrics = model.evaluate(x_test,y_test,batch_size=32, verbose=1)
    # print("\nloss:{} accuracy:{}".format(loss_and_metrics[0],loss_and_metrics[1]))

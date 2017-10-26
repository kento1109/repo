# -*- coding: utf-8-*
import re
import MeCab
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation
from keras.layers import InputLayer
from keras.callbacks import EarlyStopping, CSVLogger
from gensim.models import KeyedVectors


class ExtractCharcterFeature(object):
    def __init__(self):
        self.target_charcters = {
            "うずまきナルト": 0,
            "江戸川コナン": 1,
            "碇シンジ": 2,
            "桜木花道" :3,
            "モンキー・D・ルフィ" :4
        }

    @staticmethod
    def is_title(row):
        feature_word = "<doc"
        target_word = row[0:4]
        if target_word == feature_word:
            return True
        else:
            return False

    @staticmethod
    def is_end(row):
        feature_word = "</doc>"
        target_word = row[0:6]
        if target_word == feature_word:
            return True
        else:
            return False

    @staticmethod
    def is_target_charcter(title, target):
        if title in target.keys():
            return True
        else:
            return False

    def wakati(self, sentence):
        pos_ = ["動詞", "名詞", "形容詞"]
        # mc = MeCab.Tagger("")
        mc = MeCab.Tagger("--dicdir /opt/local/lib/mecab/dic/mecab-ipadic-neologd/")
        node = mc.parseToNode(sentence)
        wakati_list = []
        while node:
            pos = node.feature.split(",")[0]
            if pos in pos_:  # 品詞を限定
                wakati_list.append(node.surface)
            node = node.next

        return wakati_list  # list型で返す。

    def extract(self):
        is_output = False  # 出力対象行かどうか
        seq = 0  # 出力行番号
        r = re.compile(u"title=\"(?P<name>.*)\">")  # titleを抽出するための正規表現
        title = []  # title
        words = []  # sentence
        f = open('jawiki.txt', 'r')
        for row in f:
            if self.is_title(row):
                search = r.search(row)
                char_name = search.group("name")
                if self.is_target_charcter(title=char_name, target=self.target_charcters):
                    is_output = True  # 出力フラグ有効化
                    seq = 0  # 抽出行数初期化
                    continue
                else:
                    is_output = False
                    continue
                continue
            if is_output:  # Trueの場合、内容を保存
                if not self.is_end(row):
                    if seq == 0:  # title
                        char_name = row.replace('\n', '')
                    else:  # detail
                        if not row == '\n':
                            wakati_list = self.wakati(sentence=row.replace('\n', ''))
                            # １単語１データとする。
                            for word in wakati_list:
                                title.append(self.target_charcters[char_name])
                                words.append(word)
                                # print sentence
                    seq += 1

        # 変換
        title_pd = pd.Series(title)
        words_pd = pd.Series(words)
        # 結合
        df = pd.concat([title_pd, words_pd], axis=1)
        df.columns = ['label', 'word']

        return df


def sentence2sequences(ecf, maxlen, word_index, test_str):
    wakati = ecf.wakati(test_str)
    sentence = [word_index[x] for x in wakati.split(" ")]
    return pad_sequences([sentence], maxlen=maxlen, padding='post')


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


def main():
    # set parameters:
    batch_size = 32
    kernel_size = 3
    hidden_dims = 250
    epochs = 30
    num_classes = 5
    embedding_dims = 200
    # ---- 学習データ作成（ここから）----------------------------------------- #
    # ecf = ExtractCharcterFeature()
    # df = ecf.extract()
    # df.to_csv('data.csv', index=False)
    # df = pd.read_csv('data.csv', encoding='utf-8')
    #
    # # 語彙辞書の作成
    # vocab = set(df['word'])
    # vocab_dic = dict(zip(vocab, range(0, len(vocab))))
    #
    # model_bin = '../../gensim/entity_vector/entity_vector.model.bin'
    # model = KeyedVectors.load_word2vec_format(model_bin, binary=True)
    #
    # word_vectors = model.wv
    # weights = word_vectors.syn0
    # np.save(open('Word2Vec.weight', 'wb'), weights)
    #
    # # 辞書の保存
    # vocab = dict([k, v.index] for k, v in word_vectors.vocab.items())
    # with open('Word2Vec.vocab', 'w') as f:
    #     f.write(json.dumps(vocab))
    #
    # # embedding_matrixにはvocab_dicの各単語の分散表現が格納される
    # weights = np.load(open('Word2Vec.weight', 'rb'))
    # word2idx, idx2word = load_vocab()
    # embedding_matrix = np.zeros((len(vocab_dic) + 1, embedding_dims))
    # for word, i in vocab_dic.items():
    #     if word in word2idx:  # words not found in embedding index will be all-zeros.
    #         key = word2idx.get(word)
    #         embedding_matrix[i] = weights[key]
    #
    # # 学習データの作成（単語を分散表現に変換）
    # x_data = np.zeros((len(df.index) + 1, embedding_dims + 1))
    # for i, row in df.iterrows():
    #     x_data[i, 0] = row['label']  # label
    #     x_data[i, 1:] = embedding_matrix[vocab_dic[row['word']]]  # vector
    # np.save('embedded_data.npy', x_data)
    # ---- 学習データ作成（ここまで）----------------------------------------- #

    # ---- モデル作成（ここから）-------------------------------------------- #

    # # # 学習データ読み込み
    # data = np.load('embedded_data.npy')
    #
    # # ラベルをカテゴリデータに変換
    # label = to_categorical(data[:, 0])
    #
    # # データを訓練データ・検証データに分割
    # x_train, x_test, y_train, y_test = train_test_split(data[:, 1:], label, test_size=0.2, random_state=1)
    #
    # print('x_train shape:', x_train.shape)
    #
    # print('Build model...')
    # model = Sequential()
    #
    # # 入力層
    # model.add(InputLayer(input_shape=(embedding_dims,)))
    #
    # # 隠れ層
    # model.add(Dense(hidden_dims, activation='relu'))
    #
    # model.add(Dropout(0.5))
    #
    # # 隠れ層
    # model.add(Dense(hidden_dims, activation='relu'))
    #
    # model.add(Dropout(0.5))
    #
    # model.add(Dense(num_classes))
    # model.add(Activation('softmax'))
    #
    # model.compile(loss='categorical_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    #
    # print model.summary()
    #
    # callbacks = [EarlyStopping(monitor='val_loss', patience=0, verbose=1), CSVLogger("CNN_history.csv")]
    #
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(x_test, y_test),
    #           callbacks=callbacks)
    #
    # model_json_str = model.to_json()
    # open('DNN_model.json', 'w').write(model_json_str)
    # model.save_weights('DNN_weights.h5')

    # ---- モデル作成（ここまで）-------------------------------------------- #

    print "model loading"
    # モデルを読み込む
    model = model_from_json(open('DNN_model.json').read())
    # 学習結果を読み込む
    model.load_weights('DNN_weights.h5')

    weights = np.load(open('Word2Vec.weight', 'rb'))
    word2idx, idx2word = load_vocab()

    test_word = u"殺人"
    test_emd = np.zeros(embedding_dims)
    key = word2idx.get(test_word)
    if key is None:
        raise ValueError("error!")
    test_emd = weights[key]
    print model.predict(test_emd.reshape(1, embedding_dims))


if __name__ == '__main__':
    main()

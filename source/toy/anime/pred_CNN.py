# -*- coding: utf-8-*
import MeCab
import cPickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
import numpy as np

np.set_printoptions(precision=5, suppress=True)


def wakati(sentence):
    pos_ = ["動詞", "名詞", "形容詞"]
    # mc = MeCab.Tagger("--dicdir /opt/local/lib/mecab/dic/mecab-ipadic-neologd/")
    mc = MeCab.Tagger("--dicdir  /var/lib/mecab/dic/mecab-iapadic-neologd")
    node = mc.parseToNode(sentence)
    wakati_list = []
    while node:
        if node.surface:
            pos = node.feature.split(",")[0]
            if pos in pos_:  # 品詞を限定
                wakati_list.append(node.surface)
        node = node.next

    return wakati_list
    # return " ".join(wakati_list)  # str型で返す。


def sentence2sequences(maxlen, word_index, test_str):
    wakati_list = wakati(test_str)
    sentence = [word_index[x] for x in wakati_list]
    return pad_sequences([sentence], maxlen=maxlen, padding='post')


def main():
    maxlen = 600

    print "model loading"
    # モデルを読み込む
    model = model_from_json(open('CNN_model.json').read())
    # 学習結果を読み込む
    model.load_weights('CNN_weights.h5')

    # 辞書を読み込む
    word_index = cPickle.load(open("word_index.p", "rb"))

    test_data = sentence2sequences(maxlen, word_index, test_str="殺人事件に遭遇する探偵少年")
    print model.predict(test_data)

    test_data = sentence2sequences(maxlen, word_index, test_str="バスケットをする不良少年")
    print model.predict(test_data)

    test_data = sentence2sequences(maxlen, word_index, test_str="苦悩や葛藤")
    print model.predict(test_data)

    test_data = sentence2sequences(maxlen, word_index, test_str="友達")
    print model.predict(test_data)

    test_data = sentence2sequences(maxlen, word_index, test_str="成長")
    print model.predict(test_data)


if __name__ == '__main__':
    main()

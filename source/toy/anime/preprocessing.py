# -*- coding: utf-8-*
import cPickle
import re
import MeCab


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
        mc = MeCab.Tagger("--dicdir /opt/local/lib/mecab/dic/mecab-ipadic-neologd/")
        node = mc.parseToNode(sentence)
        wakati_list = []
        while node:
            pos = node.feature.split(",")[0]
            if pos in pos_:  # 品詞を限定
                wakati_list.append(node.surface)
            node = node.next

        # return wakati_list
        return " ".join(wakati_list)  # str型で返す。

    def extract(self):
        is_output = False  # 出力対象行かどうか
        seq = 0  # 出力行番号
        r = re.compile(u"title=\"(?P<name>.*)\">")  # titleを抽出するための正規表現
        title = []  # title
        sentence = []  # sentence
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
            if is_output:  # Trueの場合、内容を保存
                if not self.is_end(row):
                    if seq == 0:  # title
                        char_name = row.replace('\n', '')
                    else:  # detail
                        if not row == '\n':
                            for sent in re.split(u"。", row.replace('\n', '').decode('utf-8')):
                                wakati_str = self.wakati(sentence=sent.encode('utf-8'))
                                title.append(self.target_charcters[char_name])
                                sentence.append(wakati_str)
                            # print sentence
                    seq += 1

        return [title, sentence]


def main():
    ecf = ExtractCharcterFeature()
    label, sentence = ecf.extract()
    print len(sentence)
    # Write
    cPickle.dump((label, sentence), open("CharcterFeature.p", "wb"))


if __name__ == '__main__':
    main()

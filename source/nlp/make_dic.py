import MeCab
import csv

def read_csv(target_file):
    with open(target_file, 'r') as f:
        for row in f:
            yield row.rstrip().split(",")
    f.close()


def write_file(target_file, formats):
    with open(target_file, 'w') as f:
        for format_ in formats:
            f.write(format_)
            f.write("\n")


def make_dic_format(word, tag, mc, yomi=False, manbyo=False):
    name = word[0]
    if yomi:
        yomi = word[1]
    else:
        yomi = "*"
    if manbyo:
        tag = word[1]
    node = mc.parseToNode(name)
    while node:
        node = node.next
        format_ = ""
        if node.surface == name:  # 単語が存在する
            feature = node.feature.split(",")
            format_ += node.surface + ","  # 表層形
            format_ += str(node.lcAttr) + ","  # 左文脈ID
            format_ += str(node.rcAttr) + ","  # 右文脈ID
            if name.encode('utf-8').isalnum():
                format_ += "1,"   # 英単語は「1」で固定
            else:
                format_ += str(node.wcost - 1) + ","  # 単語の生起コスト(既存コスト-1)
            format_ += "名詞,"  # 品詞
            format_ += "一般,"  # 品詞細分類1
            format_ += "*,"  # 品詞細分類2
            format_ += "*,"  # 品詞細分類3
            format_ += feature[4] + ","  # 活用形
            format_ += feature[5] + ","  # 活用型
            format_ += tag + ","  if manbyo  else feature[6] + tag + "," # 原形
            format_ += feature[7] + "," if len(feature) > 7 else ","  # 読み
            format_ += feature[8] if len(feature) > 8 else ""  # 発音
            return format_
        else:  # 単語が存在しない（未知語）
            format_ += name + ","  # 表層形
            format_ += ","  # 左文脈ID
            format_ += ","  # 右文脈ID
            format_ += "1,"  # 未知語は「1」で固定
            format_ += "名詞,"  # 品詞
            format_ += "一般,"  # 品詞細分類1
            format_ += "*,"  # 品詞細分類2
            format_ += "*,"  # 品詞細分類3
            format_ += "*,"  # 活用形
            format_ += "*,"  # 活用型
            format_ += tag + ","  if manbyo  else feature[6] + tag + "," # 原形
            format_ += yomi + ","  # 読み
            format_ += yomi  # 発音
            return format_


parts = [part for part in read_csv(target_file="../dic/manbyo_seed.csv")]

mc = MeCab.Tagger("")
formats = [make_dic_format(word=word, tag=";part", mc=mc, manbyo=True) for word in parts]

write_file(target_file="../dic/manbyo.txt", formats=formats)
print('completed ..')


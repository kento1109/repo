# coding: UTF-8
## NHKWORLD NEWS 自動スクレイピング #####################

import urllib2
from datetime import datetime
from bs4 import BeautifulSoup

## 初期定義 #############################################
baseurl = "https://www3.nhk.or.jp/nhkworld/en/news/"
date = datetime.now()
date_str = date.strftime("%Y%m%d")
basedir = 'D:/nhk'
csvfilName = '/nhk' + date_str + '.txt'
# 改行文字数
nline = 125
# URL検索乱数開始値
iurlFrom = 1
# URL検索乱数終了値
iurlTo = 30
# 最大ページ数
maxPage = 3
#########################################################

## lineForEng ###########################################

def lineForEng(i,pletter):
  if i == 0 and pletter == ' ':
    return ('',False)
  elif i < nline:
    return (pletter,False)
  else:
    if pletter == ' ':
      return ('\n',True)
    elif pletter == '.':
      return (pletter + '\n',True)
    else:
      return (pletter,False)

## isExistURL ############################################

def isExistURL(url):
    try:
      html = urllib2.urlopen(url)
      return html
    except urllib2.HTTPError:
      return "ERR"

#########################################################

if __name__ == '__main__':

  # URLをランダムで検索
  page_num = 0
  while iurlFrom < iurlTo and page_num < maxPage:
    # アクセスするURL文を構築
    url = baseurl + date_str + "_" + str(iurlFrom).zfill(2) + "/"
    print url
    # URLの存在チェック
    html = isExistURL(url)
    if not html == "ERR":
      # htmlをBeautifulSoupで扱う
      soup = BeautifulSoup(html, "html.parser")
      #ptxtlines = soup.find_all("p",class_="PrinterFriendlyP")
      ptxtlines = soup.find_all("p")
      print soup.title.string
      # ファイルオープン
      f = open(basedir + csvfilName, 'a')
      f.write('Title:')
      f.write(soup.title.string)
      f.write('\n'*2)
      # 内容を読み込んで、textファイルに出力
      for plines in ptxtlines:
        pword = list(plines.text.replace('\n',''))
        print plines.text
        i = 0
        for pletter in pword:
          pstr,lineFlg = lineForEng(i,pletter)
          f.write(pstr)
          #改行時は文字数カウンタをリセット
          if lineFlg:
            i = 0
          else:
            i = i + 1
        f.write('\n')
      # ファイルクローズ
      # f.write('\n')
      f.write('-'*nline)
      f.write('\n'*2)
      f.close()
      page_num = page_num + 1
    iurlFrom = iurlFrom + 1
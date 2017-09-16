# coding: UTF-8
## NHKWORLD NEWS 自動スクレイピング #####################

import urllib2
import re
from datetime import datetime
from bs4 import BeautifulSoup

## 初期定義 #############################################
baseurl = "https://www.newscientist.com/article/mg23431231-000-attempts-to-alter-the-way-we-perceive-the-world/"
#baseurl = "https://www.sciencenews.org/article/fox-experiment-replaying-domestication-fast-forward"
date = datetime.now()
date_str = date.strftime("%Y%m%d")
basedir = 'D:/science'
csvfilName = '/topic' + date_str + '.txt'
# 改行文字数
nline = 125

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

    # アクセスするURL文を印字
    print baseurl
    # URLの存在チェック
    html = isExistURL(baseurl)
    if not html == "ERR":
      # htmlをBeautifulSoupで扱う
      soup = BeautifulSoup(html, "html.parser")
      #ptxtlines = soup.find_all("p",class_="PrinterFriendlyP")
      #ptxtlines = soup.find_all("p", class_=re.compile("^(?!.*published-date).+$"))
      ptxtlines = soup.find_all("p", class_="")
      print soup.title.string
      print ptxtlines
      # ファイルオープン
      #f = open(basedir + csvfilName, 'a')
      #f.write('Title:')
      #.write(soup.title.string)
      #f.write('\n'*2)
      # 内容を読み込んで、textファイルに出力
      #for plines in ptxtlines:
      for (n, plines) in enumerate(ptxtlines):
        #if n < 8:
        #    continue
        pword = list(plines.text.replace('\n',''))
        #print plines.text.encode('utf-8')
        i = 0
        for pletter in pword:
          pstr,lineFlg = lineForEng(i,pletter)
          #f.write(pstr.encode('utf-8'))
          #改行時は文字数カウンタをリセット
          if lineFlg:
            i = 0
          else:
            i = i + 1
        #f.write('\n')
      # ファイルクローズ
      #f.write('\n')
      #f.close()
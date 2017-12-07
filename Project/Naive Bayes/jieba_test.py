#coding:utf-8

import jieba

a = '今日话题'
b = jieba.cut(a)

for w in b:
    print(w.encode('utf-8'))

jieba.load_userdict("./user-dict.txt")
b = jieba.cut(a)

for w in b:
    print(w.encode('utf-8'))

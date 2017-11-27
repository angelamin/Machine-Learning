#coding:utf-8
from gensim.models.word2vec import Word2Vec
import pandas as pd
'''
加载模型
'''
model_path = './word2vec/word2vec_wx'
model = Word2Vec.load(model_path)
# sim1 = pd.Series(model.most_similar(u'微信'))
# sim2 = model.similarity(u'男朋友', u'女朋友')
# sim4 = model.similarity(u'计算机', u'老太太')
# print(sim1)
# print(sim2)
# print(sim4)
#
# list = [u'纽约', u'北京', u'美国', u'西安']
# print model.doesnt_match(list)
# list = [u'纽约', u'北京', u'华盛顿', u'女神']
# print model.doesnt_match(list)
#
# sim5 = model.wv['computer']
# print(len(sim5))
# print(sim5)
#
# print('1111')
# sim6 = model.wv[u'你好']
# print(len(sim6))
# print(sim6)
#
# print('1111')
print(model.wv.__contains__(u'全脑'))
print(model.wv.__contains__(u'你'))
tmp = model.wv.most_similar(u'你')
# print(tmp)
sim6 = model.wv[u'你']
print(type(sim6))
print(len(sim6))
print(sim6)

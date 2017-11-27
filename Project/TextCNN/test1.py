#coding:utf-8
import jieba
stopwords = {}.fromkeys([ line.rstrip() for line in open('stop_word.txt') ])
print(type(stopwords))
for x in stopwords:
    # x =x.encode('utf-8')
    unicode(x, 'utf-8')
    print(x)
stopwords = {}.fromkeys(['的', '附近','，','。'])
segs = list(jieba.cut('北京附近的租房，在这里。'))
final = ''
for seg in segs:
    seg = seg.encode('utf-8')
    if seg not in stopwords:
            final += seg + ' '
print final
print(type(final))

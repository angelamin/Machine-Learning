#coding:utf-8
import re
import jieba
def jieba_cut(inputFile,outputFile,stopwords_set,max_line_length):
    '''
    :param inputFile: 要分词的语料库
    :param outputFile:
    :return:
    '''
    fin = open(inputFile,'r')
    fout = open(outputFile,'w')
    print(stopwords_set)

    for eachLine in fin:
        line = eachLine.strip().decode('utf-8','ignore')
        if line == '':
            print('空的')
            continue
        line = re.sub('<.*?>','',line)#使用正则表达式去除html标签
        wordlist = list(jieba.cut(line, cut_all=False))#每行进行分词
        outStr = ''

        i=0
        for word in wordlist:
            # word = word.decode('utf-8')
            word = word.encode('utf-8')

            if word not in stopwords_set and not word.isdigit():
                outStr += word
                outStr += ' '
                i=i+1
        if outStr.strip() == '':
            print('空行')
        elif i>max_line_length:
            print i
        else:
            fout.write(outStr.strip()+'\n')#写入到文件中

    fin.close()
    fout.close()

def MakeWordsSet(words_file):
    print('读取文件内容......')
    words_set = set()
    with open(words_file,'r') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) >0:
                words_set.add(word)
    return words_set

if __name__ == '__main__':
    # inputFile = './mail_data/data_positive_str.txt'
    # outputFile = './mail_data/split/data_positive.txt'

    # inputFile = './mail_data/data_negative.txt'
    # outputFile = './mail_data/split/data_negative.txt'

    # inputFile = './mail_data/test_positive.txt'
    # outputFile = './mail_data/split/test_positive.txt'

    # inputFile = './mail_data/test_negative.txt'
    # outputFile = './mail_data/split/test_negative.txt'

    # inputFile = './ads/ad'
    # outputFile = './ads/split/ad'

    inputFile = './ads/not_ad'
    outputFile = './ads/split/not_ad'

    stopwords_file = './stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    max_line_length = 63   #限制每行数据的最大长度，超过的舍去

    jieba_cut(inputFile,outputFile,stopwords_set,max_line_length)

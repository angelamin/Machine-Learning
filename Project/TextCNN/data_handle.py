#coding:utf-8
import os
import chardet
import codecs

def convert_file_to_utf8(filename):
    # !!! does not backup the origin file
    content = codecs.open(filename, 'r').read()
    source_encoding = chardet.detect(content)['encoding']
    if source_encoding == None:
        print "??",filename
        return
    print "  ",source_encoding, filename
    if source_encoding != 'utf-8' and source_encoding != 'UTF-8-SIG':
        content = content.decode(source_encoding, 'ignore') #.encode(source_encoding)
        codecs.open(filename, 'w', encoding='UTF-8-SIG').write(content)

def read():
    result_file = open('/Users/xiamin/Desktop/TextCNN/mail_data/data_positive.txt','a')
    path = '/Users/xiamin/Desktop/spam'
    files = os.listdir(path)
    for file in files:#读一个文件
        print('reading'+path + '/' + file + '.......')
        convert_file_to_utf8(path + '/' + file)
        f = open(path + '/' + file)
        iter_f = iter(f)
        for line in iter_f:
            line = line.strip()
            c = line.decode('utf-8') #将str解码为unicode
            for word in c:
                if word >= u'\u4e00' and word<=u'\u9fa5':
                    result_file.write(line)
                    result_file.write('\n')
                    break

if __name__ == '__main__':
    read()

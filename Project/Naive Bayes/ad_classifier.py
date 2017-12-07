#coding:UTF-8
import cPickle
import jieba
import ast
from flask import Flask, request, jsonify
import json
from multiprocessing import Pool
import threading
import os
import time
import logging
from logging.handlers import TimedRotatingFileHandler

PROCESS_NUM = 5
MODEL_PATH = './model/ad_classifier-11-14.pkl'
WORDS_PATH = './model/feature_words-11-14.txt'
USER_DCICT_PATH = "./user-dict.txt"
jieba.load_userdict(USER_DCICT_PATH)


app = Flask(__name__)
log_file_handler = TimedRotatingFileHandler(filename="/data1/Project/shixi_xiamin/log/ad_classifier.log", when="D", interval=2, backupCount=2)
# log_file_handler = TimedRotatingFileHandler(filename="./log/porn_image.log", when="D", interval=2, backupCount=2)
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler.setFormatter(formatter)
log_file_handler.suffix = "%Y-%m-%d"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
log.addHandler(log_file_handler)

#预加载模型
log.info('开始加载模型..........')
with open(MODEL_PATH,'rb') as fid:
    classifier = cPickle.load(fid)
log.info('加载模型完毕..........')

#读取feature_words
log.info('开始加载特征........')
feature_words_file = open(WORDS_PATH,'r')
for line in feature_words_file:
    feature_words = ast.literal_eval(line)
log.info('加载特征完毕........')

@app.route('/adFilter', methods=['POST'])
def adFilter():
    try:
        results = []

        #初始化每个线程存储的数组
        names = globals()
        for i in range(0,PROCESS_NUM):
            names['result%s' % i] = []
        for j in range(0,PROCESS_NUM):
            names['input%s' % j] = []

        if request.method == 'POST':
            log.info('接收post请求........')
            txt_lines_str = request.form['txt_lines']
            txt_lines = ast.literal_eval(txt_lines_str)
            txt_lines_len = len(txt_lines)

            batch_size = txt_lines_len/PROCESS_NUM
            for m in range(0,PROCESS_NUM-1):
                for n in range(0,batch_size):
                    index = m*batch_size + n
                    input = names['input%s' % m]
                    input.append(txt_lines[index])
                # print('input')
                # for tmp in input:
                #     print(tmp)
            for k in range(index+1,txt_lines_len):
                tmp = PROCESS_NUM-1
                input = names['input%s' % tmp]
                input.append(txt_lines[k])
            # print('input----')
            # for tmp in input:
            #     print(tmp)

        threads = []
        for t in range(0,PROCESS_NUM):
            if len(names['input%s' % t]) > 0:
                t = threading.Thread(target=detect_line,args=(names['input%s' % t],names['result%s' % t]))
                threads.append(t)

        for thr in threads:
            try:
                thr.start()
            except:
                print "Error: unable to open a new thread"
        for thr in threads:
            if thr.isAlive():
                thr.join()

        for u in range(0,PROCESS_NUM):
            results.extend(names['result%s' % u])

        res = {
            'status':True,
            'msg':results
        }
        # log.info('results')
        # log.info(results)


    except Exception,e:
        log.info(e.message)
        res = {
            'status':False,
            'msg':e.message
        }

    return jsonify(res)

def detect_line(txt_lines,results):
    for line in txt_lines:
        if not line:
            results.append(-1)
        else:
            data_feature_list = TextFeatures(line,feature_words)
            result,result1,result2 = TextClassifing(classifier,data_feature_list)
            results.append(result2[0][0])

'''
函数功能：将需要分类的数据根据特征集进行向量化
Returns: 向量化的结果
'''
def TextFeatures(data,feature_words):
        data = data.strip()
        data = data.replace(' ','') #去除空格
        data_list = []
        word_cut = jieba.cut(data)
        data_list = list(word_cut)
        def text_features(text, feature_words):                                         #出现在特征集中，则置1
                text_words = set(text)
                features = [1 if word in text_words else 0 for word in feature_words]
                return features
        data_feature_list = [text_features(data_list, feature_words)]
        return data_feature_list                                #返回结果

def TextClassifing(classifier, data_feature_list):
        result = classifier.predict(data_feature_list)
        result1 = classifier.predict_log_proba(data_feature_list)
        result2 = classifier.predict_proba(data_feature_list)

        return result,result1,result2

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

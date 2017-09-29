#!/usr/bin/python
#coding:utf-8
from flask import Flask, request, jsonify
import logging
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import urllib
import ast
import numpy as np
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Pool
import threading
import json

app = Flask(__name__)

detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)


log_file_handler = TimedRotatingFileHandler(filename="./log/porn_image.log", when="D", interval=2, backupCount=2)
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler.setFormatter(formatter)
log_file_handler.suffix = "%Y-%m-%d"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
log.addHandler(log_file_handler)

@app.route('/mtcnn', methods=['POST','GET'])
def getLocation():
    process_num = 5
    results = []
    names = globals()
    for i in range(1,process_num+1):
        names['result%s' % i] = []
    for j in range(1,process_num+1):
        names['input%s' % j] = []
    if request.method == 'POST':
        image_url_str = request.form['img_urls']
        image_urls = ast.literal_eval(image_url_str)
        image_len = len(image_urls)
        for index in range(0,image_len):
            for k in range(0,process_num):
                if index%process_num == k:
                    n = k+1
                    input = names['input%s' % n]
                    input.append(image_urls[index])
        #多进程
        # print 'Parent process %s.' % os.getpid()
        # p = Pool()
        # for t in range(1,process_num+1):
        #     if len(names['input%s' % t]) > 0:
        #         p.apply_async(detect_face, args=(names['input%s' % t],names['result%s' % t]))
        # p.close()
        # p.join()

        # 多线程方法
        threads=[]
        for t in range(1,process_num+1):
            if len(names['input%s' % t]) > 0:
                t=threading.Thread(target=detect_face,args=(names['input%s' % t],names['result%s' % t]))
                threads.append(t)
        for thr in threads:
            try:
                thr.start()
            except:
                print "Error: unable to open a new thread"
        for thr in threads:
            if thr.isAlive():
                thr.join()

        for u in range(1,process_num+1):
            results.extend(names['result%s' % u])
        #将得到的结果写入文件
        with open("./data.json","w") as f:
            json.dump(results,f)
        log.info("加载入文件完成...")
        return jsonify(results)

def detect_face(image_urls,results):
    for index in range(len(image_urls)):
        img_url = image_urls[index]
        print img_url
        if img_url.startswith('http') or img_url.startswith('ftp'):
            try:
                log.info('downloading picture: ' + img_url + '.....')
                resp = urllib.urlopen(img_url)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            except:
                msg = 'error:download picture error'
                log.info(msg + 'img_url' + img_url)
                tmp = {'status':'failed-1',
                       'error_msg':msg,
                       'error_url':img_url
                      }
                results.append(tmp)
            try:
                log.info('detecting face .....')
                # start = time()
                start = time.clock()
                result = detector.detect_face(image)
                elapsed = (time.clock() - start)
                print("Time used:",elapsed)
                # stop = time()
                # print ('运行时间：' + str(stop-start) + 's')
            except Exception,e:
                msg = 'error:get face features error'
                log.info(msg + 'img data:' + str(image) + 'img_url' + img_url)
                tmp = {'status':'failed-2',
                       'error_msg':msg,
                       'error_url':img_url
                      }
                results.append(tmp)

            if 'result' in dir():
                if not result is None:
                    total_boxes = result[0].tolist()
                    points = result[1].tolist()

                    # chips = detector.extract_image_chips(image, points, 144, 0.37)

                    tmp = {'status': 'success',
                        'type':'url',
                        'total_boxes': total_boxes,
                        'points': points,
                        'url':img_url,
                        # 'chips':chips[0].tolist(),
                        'elapsed':elapsed
                        }
                    results.append(tmp)
                    log.info('detect success!')
                else:
                    tmp = {'status': 'none',
                        'url':img_url,
                        'type':'url'
                        }
                    results.append(tmp)
                    log.info('detect success!')
        else:
            try:
                log.info('reading local picture: ' + img_url + '.....')
                img = cv2.imread(img_url)
            except:
                msg = 'error:local picture error'
                log.info(msg + 'img data:' + str(img) + 'img_loaction' + img_url)
                tmp = {'status':'failed-3',
                       'error_msg':msg,
                       'error_url':img_url
                      }
                results.append(tmp)
            try:
                log.info('detecting face .....')
                result = detector.detect_face(img)
            except Exception,e:
                msg = 'error:get face features error'
                log.info(msg + 'img data:' + str(img) + 'img_url' + img_url)
                tmp = {'status':'failed-2',
                       'error_msg':msg,
                       'error_url':img_url
                      }
                results.append(tmp)

            if 'result' in dir():
                if not result is None:
                    total_boxes = result[0].tolist()
                    points = result[1].tolist()

                    #检测到的所有人脸单独照片
                    # chips = detector.extract_image_chips(img, points, 144, 0.37)

                    tmp = {'status': 'success',
                        'type':'local',
                        'total_boxes': total_boxes,
                        'points': points,
                        'url':img_url
                        # 'chips':chips[0].tolist()
                        }
                    results.append(tmp)
                    log.info('detect success!')
                else:
                    #没有检测到人脸，或者没有下载到图片
                    tmp = {'status': 'none',
                           'url':img_url,
                           'type':'local'
                        }
                    results.append(tmp)
                    log.info('detect success!')
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)

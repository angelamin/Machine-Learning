#coding:utf-8
#!/usr/bin/env python2.7
from __future__ import print_function
import sys
import threading
import numpy as np
from grpc.beta import implementations
import numpy
import tensorflow as tf
from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import json
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import time
import ast
import random

from flask import Flask,request,jsonify

app = Flask(__name__)
log_file_handler = TimedRotatingFileHandler(filename="./log/porn_image.log", when="D", interval=2, backupCount=2)
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler.setFormatter(formatter)
log_file_handler.suffix = "%Y-%m-%d"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
log.addHandler(log_file_handler)

tf.app.flags.DEFINE_integer('concurrency', 5,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/mnt', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS
PROCESS_NUM = 5
POINT = 0.5
SERVER_POOL = ['10.39.15.87:9000','10.39.15.87:9000','10.39.15.87:9000']

@app.route('/object-detection',methods=['POST','GET'])
def getLocation():
    results = []
    names = globals()
    for i in range(1,PROCESS_NUM+1):
        names['result%s' % i] = []
    for j in range(1,PROCESS_NUM+1):
        names['input%s' % j] = []
    if request.method == 'POST':
        image_url_str = request.form['img_urls']
        image_urls = ast.literal_eval(image_url_str)
        image_len = len(image_urls)
        for index in range(0,image_len):
            for k in range(0,PROCESS_NUM):
                if index%PROCESS_NUM == k:
                    n = k+1
                    input = names['input%s' % n]
                    input.append(image_urls[index])
    #多线程
    threads = []
    server_len = len(SERVER_POOL) - 1
    server_choosed = random.randint(0,server_len)
    server_address = SERVER_POOL[server_choosed]
    log.info("address")
    log.info(server_address)
    for t in range(1,PROCESS_NUM+1):
        if len(names['input%s' % t]) > 0:
            t = threading.Thread(target=do_inference,args=(server_address, FLAGS.work_dir, FLAGS.concurrency,names['input%s' % t],names['result%s' % t]))
            threads.append(t)
    for thr in threads:
        try:
            thr.start()
        except:
            print("Error: unable to open a new thread")
    for thr in threads:
        if thr.isAlive():
            thr.join()

    for u in range(1,PROCESS_NUM+1):
        results.extend(names['result%s' % u])
    return jsonify(results)
class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, concurrency):
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      log.info("error...")
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      return self._error

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1
def _create_rpc_callback( result_counter,results):
  """Creates RPC callback function.
  Args:
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.
    Calculates the statistics for the prediction result.
    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      log.info('the exception of result: '+ str(exception))
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      constant_output_boxes = result_future.result().outputs['constant_output_boxes']
      constant_output_scores = result_future.result().outputs['constant_output_scores']
      constant_output_classes = result_future.result().outputs['constant_output_classes']
      constant_output_num_detections = result_future.result().outputs['constant_output_num_detections']

      #获取得到的结果数据
      constant_output_boxes_tmp = constant_output_boxes.ListFields().pop()[1]
      constant_output_scores_tmp = constant_output_scores.ListFields().pop()[1]
      constant_output_classes_tmp = constant_output_classes.ListFields().pop()[1]

      output_scores = []
      output_scores_tmp = []
      for i in range(constant_output_scores_tmp.__len__()):
          score = constant_output_scores_tmp.pop()
          output_scores_tmp.append(score)
      #log.info('output_scores_tmp')
      #log.info(output_scores_tmp)

      output_boxes = []
      output_boxes_tmp = []
      for j in range(constant_output_boxes_tmp.__len__()):
          box = constant_output_boxes_tmp.pop()
          output_boxes_tmp.append(box)
      #log.info('output_boxes_tmp')
      #log.info(output_boxes_tmp)

      output_classes = []
      output_classes_tmp = []
      for k in range(constant_output_classes_tmp.__len__()):
          classes = constant_output_classes_tmp.pop()
          output_classes_tmp.append(classes)
     # log.info('output_classes_tmp')
      #log.info(output_classes_tmp)

      #数据过滤
      record_loc = []
      for m in range(len(output_scores_tmp)):
          if output_scores_tmp[m] >=POINT:
                record_loc.append(m)
                output_scores.append(output_scores_tmp[m])
      #log.info('output_scores')
      #log.info(output_scores)

      for n in record_loc:
        box_temp = []
        loc = n*4
        for n1 in range(4):
                loc1 = loc + n1
                box_temp.append(output_boxes_tmp[loc1])
        output_boxes.append(box_temp)
      #log.info('output_boxes')
      #log.info(output_boxes)

      for t in record_loc:
        output_classes.append(output_classes_tmp[t])
      #log.info('output_classes')
      #log.info(output_classes)

      #数据格式转换
      output_scores = np.array([output_scores])
      output_boxes = np.array([output_boxes])
      output_classes = np.array([output_classes])

      result_num_detections = []
      result_num_detections.append(len(record_loc))
      output_num_detections = np.array(result_num_detections)

      tmp = { 'constant_output_boxes':str(output_boxes),
                'constant_output_scores':str(output_scores),
                'constant_output_classes':str(output_classes),
                'constant_output_num_detections':str(output_num_detections),
      }
      results.append(tmp)
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback

def load_image_into_numpy_array2(image):
  (im_width, im_height) = image.size
  return np.asarray(image).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def do_inference(hostport, work_dir,concurrency,image_urls,results):
  """Tests PredictionService with concurrent requests.
  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
  Returns:
    The classification error rate.
  Raises:
    IOError: An error occurred processing test data set.
  """
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  result_counter = _ResultCounter(concurrency)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'objectdetection'
  request.model_spec.signature_name = 'serving_default'

  results_tmp = []
  for index in range(len(image_urls)):
      img_path = image_urls[index]
      log.info('open image.....')
      img = Image.open(img_path)
      log.info('converting image into numpy array....')
      image_np = load_image_into_numpy_array2(img)
      image_np_expanded = np.expand_dims(image_np, axis=0)
      request.inputs['constant_input_image'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image_np_expanded))
      #   result_counter.throttle()
      log.info('predicting....')
      result_future = stub.Predict.future(request, 3.0)
    #   result_future = stub.Predict(request, 3.0)
      results_tmp.append(result_future)

  log.info('predicting end....')
  time.sleep(4)
  for result_future in range(len(results_tmp)):
      results_tmp[result_future].add_done_callback(
        _create_rpc_callback(result_counter,results))

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8088, debug=True)

#coding:utf-8
#!/usr/bin/env python2.7
"""A client that talks to tensorflow_model_server loaded with object detection model.
The client  test images is local, and queries the service with
such test images to get predictions,
Typical usage example:
    python client.py --server=localhost:9000
"""
from __future__ import print_function
import sys
import threading
import numpy as np
# This is a placeholder for a Google-internal import.
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

#xiamin
@app.route('/object-detection',methods=['POST','GET'])
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

    #多线程
    threads = []
    if not FLAGS.server:
      log.info('please specify server host:port')
      return
    for t in range(1,process_num+1):
        if len(names['input%s' % t]) > 0:
            t = threading.Thread(target=do_inference,args=(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency,names['input%s' % t],names['result%s' % t]))
            threads.append(t)
    for thr in threads:
        try:
            thr.start()
        except:
            print("Error: unable to open a new thread")
    for thr in threads:
        if thr.isAlive():
            thr.join()

    for u in range(1,process_num+1):
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
      #打印出结果
      constant_output_boxes = result_future.result().outputs['constant_output_boxes']
      constant_output_scores = result_future.result().outputs['constant_output_scores']
      constant_output_classes = result_future.result().outputs['constant_output_classes']
      constant_output_num_detections = result_future.result().outputs['constant_output_num_detections']

    #   results = []
      tmp = { 'constant_output_boxes':str(constant_output_boxes),
                'constant_output_scores':str(constant_output_scores),
                'constant_output_classes':str(constant_output_classes),
                'constant_output_num_detections':str(constant_output_num_detections),
      }
      print('============')
      #print(str(constant_output_boxes))
      results.append(tmp)
      log.info('results')
    #   with open("./data.json","w") as f:
    #       json.dump(results,f)
    #   log.info("写入文件完成。。。")
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
  #test image
  for index in range(len(image_urls)):
      img_path = image_urls[index]
      print(img_path)
      log.info('open image.....')
      img = Image.open(img_path)
      log.info('converting image into numpy array....')
      image_np = load_image_into_numpy_array2(img)
      image_np_expanded = np.expand_dims(image_np, axis=0)
    #   log.info(type(image_np_expanded))
    #   print(image_np_expanded.shape)
      request.inputs['constant_input_image'].CopyFrom(
      tf.contrib.util.make_tensor_proto(image_np_expanded))
      #   result_counter.throttle()
      log.info('predicting....')
      result_future = stub.Predict.future(request, 10.0)
    #   result_future = stub.Predict(request, 3.0)
      log.info(result_future)
      results_tmp.append(result_future)

  log.info('predicting end....')
  time.sleep(10)
  for result_future in range(len(results_tmp)):
      results_tmp[result_future].add_done_callback(
        _create_rpc_callback(result_counter,results))
  # return result_counter.get_error_rate()
  # return results

# def main(_):

  # error_rate = do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency)
  # log.info('\nInference error rate: %s%%' % (error_rate * 100))

if __name__ == '__main__':
  # tf.app.run()
  app.run(host='127.0.0.1', port=8088, debug=True)

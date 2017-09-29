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


log_file_handler = TimedRotatingFileHandler(filename="./log/porn_image.log", when="D", interval=2, backupCount=2)
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler.setFormatter(formatter)
log_file_handler.suffix = "%Y-%m-%d"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
log.addHandler(log_file_handler)


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/mnt', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

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

def _create_rpc_callback( result_counter):
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
      log.info('the exception of result: '+ exception)
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      #打印出结果
      constant_output_boxes = result_future.result().outputs['constant_output_boxes']
      constant_output_scores = result_future.result().outputs['constant_output_scores']
      constant_output_classes = result_future.result().outputs['constant_output_classes']
      constant_output_num_detections = result_future.result().outputs['constant_output_num_detections']

      results = []
      tmp = { 'constant_output_boxes':str(constant_output_boxes),
                'constant_output_scores':str(constant_output_scores),
                'constant_output_classes':str(constant_output_classes),
                'constant_output_num_detections':str(constant_output_num_detections),
      }
      results.append(tmp)
      with open("./data.json","w") as f:
          json.dump(results,f)
      log.info("写入文件完成。。。")
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def do_inference(hostport, work_dir,concurrency):
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

  #test image
  image = './test_images/test.jpg'
  log.info('open image.....')
  img = Image.open(image)
  log.info('converting image into numpy array....')
  image_np = load_image_into_numpy_array(img)
  image_np_expanded = np.expand_dims(image_np, axis=0)

  request.inputs['constant_input_image'].CopyFrom(
    tf.contrib.util.make_tensor_proto(image_np_expanded, shape=[1, image_np_expanded.size]))
  result_counter.throttle()
  log.info('predicting....')
  result_future = stub.Predict.future(request, 5.0)
  log.info('predicting end....')
  result_future.add_done_callback(
        _create_rpc_callback(result_counter))
  return result_counter.get_error_rate()

def main(_):
  if not FLAGS.server:
    log.info('please specify server host:port')
    return
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency)
  log.info('\nInference error rate: %s%%' % (error_rate * 100))

if __name__ == '__main__':
  tf.app.run()

#coding:utf-8
# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""A client that talks to tensorflow_model_server loaded with mnist model.
The client downloads test images of mnist data set, queries the service with
such test images to get predictions, and calculates the inference error rate.
Typical usage example:
    mnist_client.py  --server=localhost:9000
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
import object_detection_input_data


tf.app.flags.DEFINE_integer('concurrency', 1,
                            'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
#xiamin
tf.app.flags.DEFINE_string('work_dir', '/mnt', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


# class _ResultCounter(object):
#   """Counter for the prediction results."""
#
#   def __init__(self, concurrency):
#     self._concurrency = concurrency
#     self._error = 0
#     self._done = 0
#     self._active = 0
#     self._condition = threading.Condition()
#
#   def inc_error(self):
#     with self._condition:
#       self._error += 1
#
#   def inc_done(self):
#     with self._condition:
#       self._done += 1
#       self._condition.notify()
#
#   def dec_active(self):
#     with self._condition:
#       self._active -= 1
#       self._condition.notify()

  # def get_error_rate(self):
  #   with self._condition:
  #     return self._error
  #
  # def throttle(self):
  #   with self._condition:
  #     while self._active == self._concurrency:
  #       self._condition.wait()
  #     self._active += 1


# def _create_rpc_callback( result_counter):
#   """Creates RPC callback function.
#   Args:
#     result_counter: Counter for the prediction result.
#   Returns:
#     The callback function.
#   """
#   def _callback(result_future):
#     """Callback function.
#     Calculates the statistics for the prediction result.
#     Args:
#       result_future: Result future of the RPC.
#     """
#     exception = result_future.exception()
#     if exception:
#       result_counter.inc_error()
#       print(exception)
#     else:
#       sys.stdout.write('.')
#       sys.stdout.flush()
#
#       temp = result_future.result().outputs['constant_output_classes']
#       print("66666666666    "+str(temp))
#     result_counter.inc_done()
#     result_counter.dec_active()
#   return _callback

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def do_inference(hostport, work_dir):
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
  #xiamin
  # test_data_set = object_detection_input_data.read_data_sets(work_dir).test
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  # result_counter = _ResultCounter(concurrency)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'objectdetection'
  request.model_spec.signature_name = 'serving_default'
  #xiamin
  # image = test_data_set.next_batch(1)
  image = './test_images/test.jpg'
  img = Image.open(image)
  image_np = load_image_into_numpy_array(img)
  image_np_expanded = np.expand_dims(image_np, axis=0)
  # images = np.array(img)
  # print(images.shape)
  # # data = numpy.frombuffer(buf, dtype=numpy.uint8)
  # print(img.size)
  # cols = img.width
  # rows = img.height
  # data = images.reshape(1,rows, cols, 1)
  # # data = images.reshape(cols, rows, 1)
  # images = data
  #
  # assert images.shape[0] == labels.shape[0], (
  #   'images.shape: %s ' % (images.shape))

  # Convert shape from [num examples, rows, columns, depth]
  # to [num examples, rows*columns] (assuming depth == 1)
  # assert images.shape[3] == 1
  # images = images.reshape(images.shape[0],
  #                       images.shape[1] * images.shape[2])
  # # Convert from [0, 255] -> [0.0, 1.0].
  # print("====")
  # images = images.astype(numpy.float32)
  # images = numpy.multiply(images, 1.0 / 255.0)
  #
  #
  #
  request.inputs['constant_input_image'].CopyFrom(
    tf.contrib.util.make_tensor_proto(image_np_expanded, shape=[1, image_np_expanded.size]))
  # result_counter.throttle()
  result_future = stub.Predict.future(request, 5.0)
  temp = result_future.result().outputs['constant_output_classes']
  print("66666666666    "+str(temp))
  # result_future.add_done_callback(
  #       _create_rpc_callback(result_counter))
  # print(type(result_future))# 5 seconds
  # return result_counter.get_error_rate()


def main(_):
  if not FLAGS.server:
    print('please specify server host:port')
    return
  do_inference(FLAGS.server, FLAGS.work_dir)
  # print('\nInference error rate: %s%%' % (error_rate * 100))


if __name__ == '__main__':
  tf.app.run()

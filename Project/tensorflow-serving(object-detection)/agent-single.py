#coding:utf-8
#!/usr/bin/env python2.7
from __future__ import print_function
import sys
import numpy as np
from grpc.beta import implementations
import numpy
import tensorflow as tf
from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import json
import os
import time
import ast
import random

from flask import Flask,request,jsonify

app = Flask(__name__)

tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/mnt', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS
PROCESS_NUM = 5
POINT = 0.5
SERVER_POOL = ['10.39.15.87:9000','10.39.15.87:9000','10.39.15.87:9000']

@app.route('/object-detection',methods=['POST','GET'])
def getLocation():
    server_len = len(SERVER_POOL) - 1
    server_choosed = random.randint(0,server_len)
    server_address = SERVER_POOL[server_choosed]

    results = do_inference(server_address, FLAGS.work_dir,image_url,results)

    return jsonify(results)

def load_image_into_numpy_array2(image):
  (im_width, im_height) = image.size
  return np.asarray(image).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def do_inference(hostport, work_dir,image_url,results):
  host, port = hostport.split(':')
  channel = implementations.insecure_channel(host, int(port))
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = 'objectdetection'
  request.model_spec.signature_name = 'serving_default'

  results_tmp = []
  log.info('open image.....')
  img = Image.open(img_url)
  log.info('converting image into numpy array....')
  image_np = load_image_into_numpy_array2(img)
  image_np_expanded = np.expand_dims(image_np, axis=0)
  request.inputs['constant_input_image'].CopyFrom(
  tf.contrib.util.make_tensor_proto(image_np_expanded))
  #   result_counter.throttle()

  log.info('predicting....')
  #result_future = stub.Predict.future(request, 3.0)
  result_future = stub.Predict(request, 3.0)
  results_tmp.append(result_future)
  log.info('predicting end....')
  return results_tmp

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8088, debug=True)

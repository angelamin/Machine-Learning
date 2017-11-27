#coding=utf-8
# from classifier_server import create_facenet
import tensorflow as tf
import align
import facenet
import sys
import os
import argparse
import numpy as np
import align.detect_face
import random
from datetime import datetime
from time import sleep
import tornado.ioloop
import tornado.web
from StringIO import StringIO
from PIL import Image
import urllib2
from scipy import misc
import pickle
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from grpc.beta import implementations
import logging
import json
import time
import random
import threading
from logging.handlers import TimedRotatingFileHandler

log_file_handler = TimedRotatingFileHandler(filename="./log/face_recognize.log", when="D", interval=2, backupCount=2)
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler.setFormatter(formatter)
log_file_handler.suffix = "%Y-%m-%d"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
log.addHandler(log_file_handler)
#
# def create_facenet(sess):
#     facenet.load_model("../model/")
#     # graph
#     images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#     embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#     phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
#     # graph end
#     facenet_fun = lambda images : sess.run(embeddings, feed_dict={images_placeholder: images,phase_train_placeholder:False })
#     return facenet_fun
#
# # 加载facenet模型  获取高阶函数 facenet_fun
#
# print('加载模型facenet开始')
# with tf.Graph().as_default():
#     sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#     with sess.as_default():
#         facenet_fun = create_facenet(sess)
# print('加载facenet模型成功')
#
# def facenet_network(face_images):
#     images = load_data(face_images)
#     embedings = facenet_fun(images)
#     return embedings

# SERVER_POOL = ['10.39.15.87:9000','10.39.15.87:9000','10.39.15.87:9000']
SERVER_POOL = ['0.0.0.0:9000','0.0.0.0:9000','0.0.0.0:9000']
tf.app.flags.DEFINE_integer('concurrency', 5.0,
                            'maximum number of concurrent inference requests')
FLAGS = tf.app.flags.FLAGS

#加载mtcnn模型
print('加载mtcnn模型开始')
with tf.Graph().as_default():
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

with open("../classifier.pkl", 'rb') as infile:
    (model, class_names) = pickle.load(infile)
    print('class_names')
    print(class_names)

print('加载mtcnn模型完毕')

#获取 mtcnn的人脸数量
def crop_face2(img):

    results = []

    minsize = 10  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    random_key = np.random.randint(0, high=99999)
    # try:
    #     img = misc.imread(image_path)
    # except (IOError, ValueError, IndexError) as e:
    #     errorMessage = '{}: {}'.format(image_path, e)
    #     print(errorMessage)
    if False:
        pass
    else:
        if img.ndim < 2:
            print('Unable to align "%s"' % image_path)
            # text_file.write('%s\n' % (output_filename))
        if img.ndim == 2:
            img = facenet.to_rgb(img)

        img = img[:, :, 0:3]
        a = datetime.now()
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        b = datetime.now()
        nrof_faces = bounding_boxes.shape[0]
        if nrof_faces > 0:
            det = bounding_boxes[:, 0:4]
            det_arr = []
            img_size = np.asarray(img.shape)[0:2]
            if nrof_faces > 1:
                for i in range(nrof_faces):
                    det_arr.append(np.squeeze(det[i]))
            else:
                det_arr.append(np.squeeze(det))
            #获取人脸结果
            j = 0
            for i, det in enumerate(det_arr):
                j += 1

                det = np.squeeze(det)
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - 4 / 2, 0)
                bb[1] = np.maximum(det[1] - 4 / 2, 0)
                bb[2] = np.minimum(det[2] + 4 / 2, img_size[1])
                bb[3] = np.minimum(det[3] + 4 / 2, img_size[0])
                cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
                scaled = misc.imresize(cropped, (182, 182), interp='bilinear')
                # nrof_successfully_aligned += 1
                # output_filename_n = "{}_{}.{}".format(output_filename.split('.')[0], i, output_filename.split('.')[-1])
                # image_path.split('.')[0]
                # print('scaled')
                # print(scaled)
                results.append(scaled)
                # print('scaled')
                # print(scaled)

            return results
        else:
            return []

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image

def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image

def load_data(scales, do_random_crop=False, do_random_flip=False, image_size=160, do_prewhiten=True):
    images = np.zeros((len(scales), image_size, image_size, 3))
    for i in range(len(scales)):
        img = scales[i]
        if img.ndim == 2:
            img = to_rgb(img)
        if do_prewhiten:
            img = prewhiten(img)
        img = crop(img, do_random_crop, image_size)
        img = flip(img, do_random_flip)
        images[i,:,:,:] = img
    return images

def load_image_into_numpy_array2(image):
  (im_width, im_height) = image.size
  return np.asarray(image).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

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
def _create_rpc_callback(result_counter,results):
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
    # log.info('dir(result_future)')
    # log.info(dir(result_future))
    if exception:
      result_counter.inc_error()
      log.info('the exception of result: '+ str(exception))
    else:
      sys.stdout.write('.')
      sys.stdout.flush()
      constant_output_embedding = result_future.result().outputs['constant_output_embedding']
      log.info('constant_output_embedding')
      log.info(constant_output_embedding)
      #获取得到的结果数据
      constant_output_embedding_tmp = constant_output_embedding.ListFields().pop()[1]
      log.info('constant_output_embedding_tmp')
      log.info(constant_output_embedding_tmp)

      # output_embedding = []
      # output_embedding_tmp = []
      # for j in range(constant_output_embedding_tmp.__len__()):
      # for j in range(128):
      #     embedding = constant_output_embedding_tmp.pop()
      #     output_embedding_tmp.append(embedding)
          # print('embedding')
          # print(embedding)


      # print('output_embedding_tmp')
      # print(output_embedding_tmp)
      #数据格式转换

      # output_embedding = np.array([output_embedding])
      # output_embedding_tmp = np.array(output_embedding_tmp)
      output_embedding_tmp = np.array(constant_output_embedding_tmp)

      # results.append(output_embedding)
      results.append(output_embedding_tmp)
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback

def do_inference(hostport,face_images,results,concurrency):
    '''
    concurrency: Maximum number of concurrent requests.
    images:np array that mtcnn returns
    '''
    host,port = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    result_counter = _ResultCounter(concurrency)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'face_recognization'
    request.model_spec.signature_name = 'serving_default'

    results_tmp = []
    #遍历images(每个人脸的numpy数组)
    i = 20

    images = load_data(face_images)

    for img_np in images:
         image_np_expanded = np.expand_dims(img_np,axis=0)
         for result in image_np_expanded:
             name = str(i) + '.png'
             misc.imsave(name,result)
             i += 1
         request.inputs['constant_input_image'].CopyFrom(
            tf.contrib.util.make_tensor_proto(image_np_expanded,dtype=tf.float32))
         request.inputs['constant_input_phase_train'].CopyFrom(tf.contrib.util.make_tensor_proto(False))
         log.info('requesting server........')
         result_future = stub.Predict.future(request,3.0)
         log.info('result_future')
         log.info(result_future)
         results_tmp.append(result_future)

    log.info('requesting end......')
    time.sleep(4)
    for result_future in range(len(results_tmp)):
        results_tmp[result_future].add_done_callback(
            _create_rpc_callback(result_counter,results)
        )
class MainHandler(tornado.web.RequestHandler):
    def get(self):
        img_url = self.get_argument('img_url')
        log.info('opening image.......')
        log.info(img_url)
        r = urllib2.urlopen(img_url)
        image_data = r.read()

        buff = StringIO()
        buff.write(image_data)
        buff.seek(0)
        img = Image.open(buff)
        image_np = load_image_into_numpy_array2(img)
        log.info('begin face detecion and resize.......')
        images = crop_face2(image_np)
        log.info('face detecion and resize finished......')

        # 将图片转换为128维向量
        # 请求model server的方式
        server_len = len(SERVER_POOL) - 1
        server_choosed = random.randint(0,server_len)
        server_address = SERVER_POOL[server_choosed]

        embedings = []

        log.info('begin embedding.......')
        do_inference(server_address,images,embedings,FLAGS.concurrency)
        log.info('embedding finished......')

        # embedings = facenet_network(images)

        log.info('embedings')
        log.info(embedings)

        # file1 = open('embedings.txt','w')
        # for embedding in embedings:
        #     print(embedding)
        #     file1.write(str(embedding))

        log.info('predicting.....')
        predictions = model.predict_proba(embedings)
        print('predictions')
        print(predictions)
        best_class_indices = np.argmax(predictions,axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)),best_class_indices]

        print('分类结果')
        for i in range(len(best_class_indices)):
            print('%s : %.3f' % (class_names[best_class_indices[i]],best_class_probabilities[i]))
        return

if __name__=='__main__':
    Handlers = [
        (r"/", MainHandler),
    ]
    application = tornado.web.Application(Handlers)
    application.listen(1108)
    tornado.ioloop.IOLoop.instance().start()
    '''
    print('start------------')
    image_path = '/Users/yahui3/yahui3/facenet/shezheng_face/xijinping/1_1.png'
    images = crop_face2(image_path)
    print('检测到人脸数量为：'+str(len(images)))
    embedings = facenet_network(images)
    print(embedings)
    '''

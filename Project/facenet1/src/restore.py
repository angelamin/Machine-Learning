#coding:utf-8
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
from datetime import datetime
from collections import defaultdict
from io import StringIO
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  '../model/20170512-110547.pb'
POINT = 0.5

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        image_tensor = detection_graph.get_tensor_by_name('input:0')
        phase_train_tensor = detection_graph.get_tensor_by_name("phase_train:0")

        embeddings = detection_graph.get_tensor_by_name('embeddings:0')

        # Visualization of the results of a detection.
        # Export inference model.
        output_path = "../restored_model/1"
        print 'Exporting trained model to', output_path
        builder = saved_model_builder.SavedModelBuilder(output_path)  #初始化SavedModel

        # Build the signature_def_map.
        image_inputs_tensor_info = utils.build_tensor_info(image_tensor)      #将变量生成对应的proto buffer信息
        phase_train_inputs_tensor_info = utils.build_tensor_info(phase_train_tensor)
        embeddings_output_tensor_info = utils.build_tensor_info(embeddings)

        classification_signature = signature_def_utils.build_signature_def(
            inputs={
                'constant_input_image': image_inputs_tensor_info,
                'constant_input_phase_train':phase_train_inputs_tensor_info
            },
            outputs={
                'constant_output_embedding':embeddings_output_tensor_info
            },
            method_name=signature_constants.CLASSIFY_METHOD_NAME)

        legacy_init_op = tf.group(
            tf.initialize_all_tables(), name='legacy_init_op')   #table是一个字符串的映射表

        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature,
            },
            legacy_init_op=legacy_init_op)          #将sess和变量保存至SavedModel

        builder.save()                  #模型保存在SavedModel初始化的路径中
        print 'Successfully exported model to %s' % output_path

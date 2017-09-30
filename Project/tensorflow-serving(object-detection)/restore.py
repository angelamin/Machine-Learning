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
PATH_TO_CKPT =  'output_inference_graph.pb'
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
        #image_path = "./imgs/test.jpg"
        #image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        #image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        #image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        #a=datetime.now()
        '''
        (result_boxes, result_scores, result_classes, result_num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        '''
        # Visualization of the results of a detection.
        # Export inference model.
        output_path = "/tmp/output/1"
        print 'Exporting trained model to', output_path
        builder = saved_model_builder.SavedModelBuilder(output_path)  #初始化SavedModel

        # Build the signature_def_map.
        image_inputs_tensor_info = utils.build_tensor_info(
                        image_tensor)      #将变量生成对应的proto buffer信息
        '''
        boxes_output_tensor_info = utils.build_tensor_info(tf.convert_to_tensor(result_boxes))
        scores_output_tensor_info = utils.build_tensor_info(tf.convert_to_tensor(result_scores))
        classes_output_tensor_info = utils.build_tensor_info(tf.convert_to_tensor(result_classes))
        num_detections_output_tensor_info = utils.build_tensor_info(tf.convert_to_tensor(result_num_detections))
	'''
	
	boxes_output_tensor_info = utils.build_tensor_info(boxes)
        scores_output_tensor_info = utils.build_tensor_info(scores)
        classes_output_tensor_info = utils.build_tensor_info(classes)
        num_detections_output_tensor_info = utils.build_tensor_info(num_detections)

        classification_signature = signature_def_utils.build_signature_def(
            inputs={
                'constant_input_image': image_inputs_tensor_info
            },
            outputs={
                'constant_output_boxes':
                    boxes_output_tensor_info,
                'constant_output_scores':
                    scores_output_tensor_info,
                'constant_output_classes':
                    classes_output_tensor_info,
                'constant_output_num_detections':
                    num_detections_output_tensor_info
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

#coding:UTF-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
from lib_ctpn.networks.factory import get_network
from lib_ctpn.fast_rcnn.config import cfg,cfg_from_file
from lib_ctpn.fast_rcnn.test import test_ctpn
from lib_ctpn.utils.timer import Timer
from lib_ctpn.utils.blob import im_list_to_blob
from lib_ctpn.text_connector.detectors import TextDetector
from lib_ctpn.text_connector.text_connect_cfg import Config as TextLineCfg
from PIL import Image
from PIL import ImageFile
import imghdr
import pytesseract
import os.path as ops
import argparse
#import matplotlib.pyplot as plt
# from lib_crnn.crnn_model import crnn_model
# from lib_crnn.global_configuration import config as config_crnn
# from lib_crnn.local_utils import log_utils, data_utils
try:
    from cv2 import cv2
except ImportError:
    pass
from flask import Flask, request, jsonify
import time
import logging
from logging.handlers import TimedRotatingFileHandler
import json
import traceback
from tensorflow.python.platform import gfile
import re


app = Flask(__name__)
log_file_handler = TimedRotatingFileHandler(filename="/ctpn_crnn_in/log/text-recognization.log", when="D", interval=2, backupCount=2)
log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler.setFormatter(formatter)
log_file_handler.suffix = "%Y-%m-%d"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
log.addHandler(log_file_handler)
#加载恢复ctpn模型
def _get_image_blob(im):
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)

def get_blobs(im, rois):
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    return blobs, im_scale_factors

def create_ctpnnet(sess):
    cfg_from_file('lib_ctpn/text.yml')
    net_ctpn = get_network("VGGnet_test")
    # load_model("checkpoints_ctpn/")
    saver = tf.train.Saver()
    saver.restore(sess,'checkpoints_ctpn/VGGnet_fast_rcnn_iter_50000.ckpt')

    ctpn_fun = lambda blobs : sess.run([net_ctpn.get_output('rois')[0]],feed_dict={net_ctpn.data: blobs['data'], net_ctpn.im_info: blobs['im_info'], net_ctpn.keep_prob: 1.0})

    return ctpn_fun

log.info('begin loading model..........')

with tf.Graph().as_default():
    sess = tf.Session(config = tf.ConfigProto(allow_soft_placement=True))
    with sess.as_default():
        ctpn_fun = create_ctpnnet(sess)

log.info('model loaded..........')


def ctpn_network(blobs,im_scales,boxes=None):
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
    rois = ctpn_fun(blobs)
    print('rois')
    print(rois)
    rois=rois[0]

    scores = rois[:, 0]
    print('scores')
    print(scores)
    if cfg.TEST.HAS_RPN:
        print('has rpn........')
        assert len(im_scales) == 1, "Only single-image batch implemented"
        boxes = rois[:, 1:5] / im_scales[0]
        print('boxes......................11111111111111111111111111')
        print(boxes)
    return scores,boxes

def image_feed(im,boxes=None):
    # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)

    blobs, im_scales = get_blobs(im, boxes)
    if cfg.TEST.HAS_RPN:
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
            dtype=np.float32)
    return blobs,im_scales

@app.route('/textRecognization', methods=['POST'])
def textRecogn():
    print('11111111111111111111111111')
    results = []

    if request.method == 'POST':
        image_url = request.form['image_url']
        log.info(('detection for {:s}'.format(image_url)))
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            if imghdr.what(image_url) == "png":
                Image.open(image_url).convert("RGB").save(image_url)
            img = cv2.imread(image_url)
            print("before resize, the shape........")
            print(img.shape)
            img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
            print("before test_ctpn, the shape.........%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(img.shape)
            # im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
            blobs,im_scales = image_feed(img)
            scores, boxes = ctpn_network(blobs,im_scales)

            # scores, boxes = test_ctpn(sess, net, img)
            textdetector = TextDetector()
            boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
            print('boxes.............................~~~~~~~~~~~~~~~~~~~')
            print(boxes)
            draw_boxes(img, image_url, boxes, scale)
            # timer.toc()
            # print(('Detection took {:.3f}s for '
                   # '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
            #OCR recognition
            results = recognition(img,image_url,boxes,scale)
            print('results.....')
            print(results)
            # return results
            res = {
                'status':True,
                'msg':str(results)
            }
        except Exception as e:
            log.info(traceback.format_exc())
            res = {
                'status':False,
                'msg':traceback.format_exc()
            }
        return jsonify(res)

def resize_im(im, scale, max_scale=None):
    f=float(scale)/min(im.shape[0], im.shape[1])
    if max_scale!=None and f*max(im.shape[0], im.shape[1])>max_scale:
        f=float(max_scale)/max(im.shape[0], im.shape[1])
    return cv2.resize(im, None,None, fx=f, fy=f,interpolation=cv2.INTER_LINEAR), f

def draw_boxes(img,image_name,boxes,scale):
    for box in boxes:
        print("box........................")
        print(box)
        print(box[0])
        print(box[1])
        print(box[2])
        print(box[3])
        print(box[4])
        print(box[5])
        print(box[6])
        print(box[7])
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), (0, 255, 0), 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), (0, 255, 0), 2)

    base_name = image_name.split('/')[-1]
    img=cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join("data_ctpn/results_test", base_name), img)

def recognition(img1,image_name,boxes,scale):
    results = []
    for box in boxes:
        img = Image.fromarray(img1)
        width = img.size[0]
        height = img.size[1]
        w1 = int(width-box[2])
        w2 = int(height-box[5])
        w3 = int(width-box[0])
        w4 = int(height-box[1])
        crop_info = (w1,w2,w3,w4)
        roi = img.crop(crop_info)
        text = pytesseract.image_to_string(roi,lang='chi_sim')
        result = text.encode('utf-8')
        print('detect result...........')
        print(result)
        results.append(result)
    return results

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1889, use_reloader=False)
    # image_url = './7.jpg'
    # textRecogn(image_url)

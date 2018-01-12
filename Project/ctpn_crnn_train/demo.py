from __future__ import print_function
import tensorflow as tf
import numpy as np
import os, sys, cv2
import glob
import shutil
sys.path.append(os.getcwd())
#sys.path.append("..")
from lib_ctpn.networks.factory import get_network
from lib_ctpn.fast_rcnn.config import cfg,cfg_from_file
from lib_ctpn.fast_rcnn.test import test_ctpn
from lib_ctpn.utils.timer import Timer
from lib_ctpn.text_connector.detectors import TextDetector
from lib_ctpn.text_connector.text_connect_cfg import Config as TextLineCfg
from PIL import Image
from PIL import ImageFile
import imghdr
import pytesseract

import os.path as ops
import argparse
#import matplotlib.pyplot as plt
from lib_crnn.crnn_model import crnn_model
from lib_crnn.global_configuration import config as config_crnn
from lib_crnn.local_utils import log_utils, data_utils
try:
    from cv2 import cv2
except ImportError:
    pass

logger = log_utils.init_logger()

#from skimage import io
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
    cv2.imwrite(os.path.join("data/results_test", base_name), img)

def recognition(img1,image_name,boxes,scale):
    for box in boxes:
        print("recognize box...........")
        print(box)
       # io.imsave('./test.jpg',img)
       # cv2.imwrite('./test.jpg',img1)
       # img = Image.open('./test.jpg')
        #img = Image.open(image_name)
        img = Image.fromarray(img1)
        width = img.size[0]
        height = img.size[1]
        #height,width,channels = img.shape
        w1 = int(width-box[2])
        w2 = int(height-box[5])
        w3 = int(width-box[0])
        w4 = int(height-box[1])
        crop_info = (w1,w2,w3,w4)
        roi = img.crop(crop_info)
       # io.imsave('./test.jpg',img)
       # img = Image.open('./test.jpg')
       # box1 = int(box[1])
        #box5 = int(box[5])
       # box0 = int(box[0])
       # box2 = int(box[2])
        #roi = img[box1:box5,box0:box2]
        text = pytesseract.image_to_string(roi,lang='chi_sim')
        print("hahahahahah.........")
        print(text.encode('utf-8'))
def ctpn(sess, net, image_name):
    with sess.as_default():
        with sess.graph.as_default():
            timer = Timer()
            timer.tic()
            #dealing  the problem of reading png error
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            if imghdr.what(image_name) == "png":
                Image.open(image_name).convert("RGB").save(image_name)
            img = cv2.imread(image_name)
           # img = np.array(Image.open(image_name))
            print("before resize, the shape........")
            print(img.shape)
            img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
            print("before test_ctpn, the shape.........%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print(img.shape)
            scores, boxes = test_ctpn(sess, net, img)

            print("boxes.........")
            print(boxes)
            #print("scores.......")
            #print(scores)
            textdetector = TextDetector()
            boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
            #print("start drawing.....")
            #draw_boxes(img, image_name, boxes, scale)
            #print("drawing end......")
            timer.toc()
            print(('Detection took {:.3f}s for '
                   '{:d} object proposals').format(timer.total_time, boxes.shape[0]))
            #OCR recognition
            recognition(img,image_name,boxes,scale)

            return img,boxes

def crnn(sess,net,img,boxes):
    with sess.as_default():
        with sess.graph.as_default():
            image = cv2.resize(img, (100, 32))
            image = np.expand_dims(image, axis=0).astype(np.float32)

            preds = sess.run(decodes, feed_dict={inputdata: image})
            preds = decoder.writer.sparse_tensor_to_str(preds[0])

            logger.info('Predict image label {:s}'.format(preds[0]))

         #   if is_vis:
          #      plt.figure('CRNN Model Demo')
           #     plt.imshow(cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, (2, 1, 0)])
            #    plt.show()

    # sess.close()

if __name__ == '__main__':
   # if os.path.exists("data/results_test/"):
    #    shutil.rmtree("results/results_test/")
    #os.makedirs("results/results_test/")
    '''
    恢复crnn模型
    '''
    # # config tf session
    # sess_config = tf.ConfigProto()
    # sess_config.gpu_options.per_process_gpu_memory_fraction = config_crnn.cfg.TRAIN.GPU_MEMORY_FRACTION
    # sess_config.gpu_options.allow_growth = config_crnn.cfg.TRAIN.TF_ALLOW_GROWTH
    # g1 = tf.Graph()
    # sess_crnn = tf.Session(config=sess_config,graph=g1)
    # with sess_crnn.as_default():
    #     with g1.as_default():
    #         inputdata = tf.placeholder(dtype=tf.float32, shape=[1, 32, 100, 3], name='input')
    #         net_crnn = crnn_model.ShadowNet(phase='Test', hidden_nums=256, layers_nums=2, seq_length=25, num_classes=37)
    #
    #         with tf.variable_scope('shadow'):
    #             net_out = net_crnn.build_shadownet(inputdata=inputdata)
    #
    #         decodes, _ = tf.nn.ctc_beam_search_decoder(inputs=net_out, sequence_length=25*np.ones(1), merge_repeated=False)
    #
    #         decoder = data_utils.TextFeatureIO()
    #
    #
    #         # config tf saver
    #         print(('Loading network {:s}........................................................................ '.format("shadownet_2017")), end=' ')
    #         saver = tf.train.Saver()
    #
    #         print('1111111111111111111111111111111111111111111111111111111')
    #         # saver.restore(sess, 'checkpoints_crnn/shadownet_2017-10-17-11-47-46.ckpt')
    #         saver.restore(sess=sess_crnn, save_path='checkpoints_crnn/shadownet_2017-10-17-11-47-46.ckpt-199999')
    #
    #         print('222222222222222222222222222222222222222222222222222222222')
    #

    '''
    恢复ctpn模型
    '''
    # init session
    config = tf.ConfigProto(allow_soft_placement=True)
    g2 = tf.Graph()
    sess_ctpn = tf.Session(config=config,graph=g2)
    with sess_ctpn.as_default():
        with g2.as_default():
            cfg_from_file('lib_ctpn/text.yml')
            # load network
            net_ctpn = get_network("VGGnet_test")
            # load model
            print(('Loading network {:s}... '.format("VGGnet_test")), end=' ')
            saver = tf.train.Saver()

            try:
                ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
                #print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                #saver.restore(sess, ckpt.model_checkpoint_path)
                saver.restore(sess_ctpn,'checkpoints_ctpn/VGGnet_fast_rcnn_iter_50000.ckpt')
                print('done')
            except:
                raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

            im = 128 * np.ones((300, 300, 3), dtype=np.uint8)
            for i in range(2):
                _, _ = test_ctpn(sess_ctpn, net_ctpn, im)

    im_names = glob.glob(os.path.join(cfg.DATA_DIR, 'debug', '*.png')) + \
               glob.glob(os.path.join(cfg.DATA_DIR, 'debug', '*.jpg'))


    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('Demo for {:s}'.format(im_name)))
        img,boxes = ctpn(sess_ctpn, net_ctpn, im_name)
        # crnn(sess_crnn,net_crnn,img,boxes)

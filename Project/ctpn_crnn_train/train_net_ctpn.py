#coding:utf-8
import pprint
import sys
import os.path

sys.path.append(os.getcwd())
this_dir = os.path.dirname(__file__)

from lib_ctpn.fast_rcnn.train import get_training_roidb, train_net
from lib_ctpn.fast_rcnn.config import cfg_from_file, get_output_dir, get_log_dir
from lib_ctpn.datasets.factory import get_imdb
from lib_ctpn.networks.factory import get_network
from lib_ctpn.fast_rcnn.config import cfg

if __name__ == '__main__':
    #将text.yml中配置同步到cfg 中
    cfg_from_file('lib_ctpn/text.yml')
    print('Using config:')
    pprint.pprint(cfg)
    #获取样本
    imdb = get_imdb('voc_2007_trainval')
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    log_dir = get_log_dir(imdb)
    print('Output will be saved to `{:s}`'.format(output_dir))
    print('Logs will be saved to `{:s}`'.format(log_dir))

    device_name = '/gpu:0'
    print(device_name)

    #factory.py根据参数VGGnet_train是训练网络,得到模型的结构   VGGnet_train()
    #VGGnet_train.py 网络结构
    #network.py中包含 VGGnet_train.py中声明的函数
    network = get_network('VGGnet_train')
    #在sess中导入输入，开始模型训练，传入网络结构
    train_net(network, imdb, roidb,
              output_dir=output_dir,
              log_dir=log_dir,
              pretrained_model='/ctpn_side-refinement_in/data_ctpn/pretrain/VGG_imagenet.npy',
              max_iters=60000,
              restore=bool(int(cfg.TRAIN.restore)))

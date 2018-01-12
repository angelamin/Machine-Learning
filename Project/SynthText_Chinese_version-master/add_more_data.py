#coding:utf-8
import numpy as np
import h5py
import os, sys, traceback
import os.path as osp
import wget, tarfile
import cv2
from PIL import Image

def add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path):
  db=h5py.File(DB_FNAME,'a')
  depth_db = h5py.File(more_depth_path,'r')
  print 'depth_db.........................'
  print depth_db
  seg_db = h5py.File(more_seg_path,'r')
  print 'seg_db..........................'
  print seg_db
  # depth_db=get_data(more_depth_path)
  # seg_db=get_data(more_seg_path)

  print('db.keys()------------------------------')
  print(db.keys())

  # if 'image1' not in db.keys():
  #     db.create_group('image')
  # if 'depth1' not in db.keys():
  #     db.create_group('depth')
  # if 'seg1' not in db.keys():
  #     db.create_group('seg')

  db.create_group('image')
  db.create_group('depth')
  db.create_group('seg')

  for imname in os.listdir(more_img_file_path):
    if imname.endswith('.jpg'):
      full_path=more_img_file_path+imname
      print full_path,imname

      try:
          j=Image.open(full_path)
          imgSize=j.size
          rawData=j.tobytes()
          img=Image.frombytes('RGB',imgSize,rawData)
          print 'img........................'
          print img
          #img = img.astype('uint16')
          db['image'].create_dataset(imname,data=img)
          db['depth'].create_dataset(imname,data=depth_db[imname])
          db['seg'].create_dataset(imname,data=seg_db['mask'][imname])

          db['seg'][imname].attrs['area']=seg_db['mask'][imname].attrs['area']
          db['seg'][imname].attrs['label']=seg_db['mask'][imname].attrs['label']
      except:
          print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
          print full_path,imname
          continue
  db.close()
  depth_db.close()
  seg_db.close()


# path to the data-file, containing image, depth and segmentation:
DB_FNAME = '/Users/xiamin/Desktop/SynthText_Chinese_version-master/chinese_sytnthtext_pre_data/dset_8000_1.h5'

#add more data into the dset
more_depth_path='/Users/xiamin/Desktop/SynthText_Chinese_version-master/chinese_sytnthtext_pre_data/depth1.h5'
more_seg_path='/Users/xiamin/Desktop/SynthText_Chinese_version-master/chinese_sytnthtext_pre_data/seg_uint16_1.h5'
more_img_file_path='/Users/xiamin/Desktop/SynthText_Chinese_version-master/data/bg_img1/'

add_more_data_into_dset(DB_FNAME,more_img_file_path,more_depth_path,more_seg_path)

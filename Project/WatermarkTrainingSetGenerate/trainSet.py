#coding:utf-8
# import random
from random import randint
from os import listdir
from PIL import Image,ImageEnhance
import os
import re

def reduce_opacity(im, opacity): # 处理Logo的透明度
    assert opacity >= 0 and opacity <= 1
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    else:
        im = im.copy()
    alpha = im.split()[3]
    alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
    im.putalpha(alpha)
    return im

def addwatermark(txtdir,index,imagefile,markfile,opacity):
    im = Image.open(imagefile)
    mark = Image.open(markfile)

    im_w, im_h = im.size
    mark_w,mark_h = mark.size

    proportion = (im_w*im_h)/(mark_w*mark_h)

    if im_w < mark_w or im_h < mark_h or proportion>900:
        if proportion>900:
            print '图片太大.. 图片：'
            print imagefile
        else:
            print '水印图片size比图片大.. 图片：'
            print imagefile
        return 'none'
    else:
        if opacity < 1:
            mark = reduce_opacity(mark, opacity)
            if im.mode != 'RGBA':
                im = im.convert('RGBA')
            # 创建一个透明层，根据位置 来画Logo.
            layer = Image.new('RGBA', im.size, (0,0,0,0))

            left = randint(0,im_w - mark_w)
            top = randint(0,im_h - mark_h)
            layer.paste(mark,(left,top))

            #将位置信息存储
            location_path = './' + txtdir + '/' + str(index) + '.txt'
            f = open(location_path,'w')
            f.write('(' + str(left) + ',' + str(top) + ')' + '\n')
            f.write('(' + str(left + mark_w) + ',' + str(top) + ')' + '\n')
            f.write('(' + str(left) + ',' + str(top+mark_h) + ')' + '\n')
            f.write('(' + str(left+mark_w) + ',' + str(top+mark_h) + ')' + '\n')
            f.close()
            return Image.composite(layer, im, layer)

if __name__ == '__main__':
    MARKIMAGE = r'./watermark/zairenjian_logo.png' # Logo图片的位置

    dir = 1
    pathdir = 'marked' + str(dir)
    while os.path.exists(pathdir):
        dir = dir + 1
        pathdir = 'marked' + str(dir)

    pathdir = 'marked' + str(dir)
    os.mkdir(pathdir)
    txtdir = pathdir + '_location'
    os.mkdir(txtdir)

    srcDir = './imgs'
    opacity = 0.6
    index = 1

    picFiles = [fn for fn in listdir(srcDir) if fn.endswith(('.bmp', '.jpg', '.png'))]
    for fn in picFiles:
        img_path = srcDir + '/' + fn
        for i in range(5):
            image = addwatermark(txtdir,index,img_path,MARKIMAGE,opacity)
            print image
            if image == 'none':
                continue
            else:
                new_path = './' + pathdir + '/' + str(index) + '.png'
                image.save(new_path,quality=90)
                index = index+1

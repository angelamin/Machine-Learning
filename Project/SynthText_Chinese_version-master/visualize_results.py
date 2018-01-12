"""
Visualize the generated localization synthetic
data stored in h5 data-bases
"""
from __future__ import division
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import h5py
from common import *
from os import path
import scipy.misc

def viz_textbb(k,txt,text_im, charBB_list, wordBB, alpha=1.0):
    """
    text_im : image containing text
    charBB_list : list of 2x4xn_i bounding-box matrices
    wordBB : 2x4xm matrix of word coordinates
    """
    plt.close(1)
    plt.figure(1)
    plt.imshow(text_im)
    plt.hold(True)
    H,W = text_im.shape[:2]
    # vardict=dir()
    # if 'image_num' in vardict:
    #     pass
    # else:
    #     image_num = 0
    # plot the character-BB:
    # for i in xrange(len(charBB_list)):
    #     bbs = charBB_list[i]
    #     ni = bbs.shape[-1]
    #     for j in xrange(ni):
    #         bb = bbs[:,:,j]
    #         bb = np.c_[bb,bb[:,0]]
    #         plt.plot(bb[0,:], bb[1,:], 'r', alpha=alpha/2)


    file_sample = open('output/sample.txt','a')

    # print('****************')
    # print(txt.shape[0])
    # print(wordBB.shape[-1])
    if txt.shape[0] == wordBB.shape[-1]:
        for i in xrange(wordBB.shape[-1]):
            # print('i..........................................................')
            # print(i)
            # print(k.split('.')[0].encode('utf-8'))
            # print('txt^^^^^^^^^^^^^^^^^^^^^^')
            # print(txt[i])
            # print(txt[i].encode('utf-8'))
            image_name = k.split('.')[0].encode('utf-8')
            image_name = str(image_name) + str(i)
            bb = wordBB[:,:,i]
            bb = np.c_[bb,bb[:,0]]
            # print('bb.................')
            # print(bb.shape)
            # print(bb)
            # print('bb[0,:])........')
            # print(bb[0,:].shape)
            # print(bb[0,:])
            # print('bb[1,:]............')
            # print(bb[1,:].shape)
            # print(bb[1,:])
            plt.plot(bb[0,:], bb[1,:], 'g', alpha=alpha)
            min_x = int(min(bb[0,:]))
            min_y = int(min(bb[1,:]))
            max_x = int(max(bb[0,:]))
            max_y = int(max(bb[1,:]))
            # region = text_im.crop((min_x,min_y,max_x,max_y))
            region = text_im[min_y:max_y,min_x:max_x]
            image_file = str(image_name) +'_1'+ '.jpg'
            output_file = path.join('output/',image_file)
            # region.save(output_file)
            try:
                scipy.misc.imsave(output_file, region)
                print('saving image to ' + output_file)
                file_sample.write(output_file + ' ' + txt[i].encode('utf-8'))
                file_sample.write('\n')
            except:
                continue
            # visualize the indiv vertices:
            # vcol = ['r','g','b','k']
            # for j in xrange(4):
            #     plt.scatter(bb[0,j],bb[1,j],color=vcol[j])

        # plt.gca().set_xlim([0,W-1])
        # plt.gca().set_ylim([H-1,0])
        # plt.show(block=False)

def main(db_fname):
    db = h5py.File(db_fname, 'r')
    dsets = sorted(db['data'].keys())
    # print "total number of images : ", colorize(Color.RED, len(dsets), highlight=True)
    for k in dsets:
        rgb = db['data'][k][...]
        charBB = db['data'][k].attrs['charBB']
        wordBB = db['data'][k].attrs['wordBB']
        txt = db['data'][k].attrs['txt']

        # print "image name        : ", colorize(Color.RED, k, bold=True)
        # print "  ** no. of chars : ", colorize(Color.YELLOW, charBB.shape[-1])
        # print "  ** no. of words : ", colorize(Color.YELLOW, wordBB.shape[-1])
        # print "  ** text         : ", colorize(Color.GREEN, txt)

        viz_textbb(k,txt,rgb, [charBB], wordBB)


        # if 'q' in raw_input("next? ('q' to exit) : "):
        #     break
    db.close()

if __name__=='__main__':
    main('results/SynthText_8000_1.h5')

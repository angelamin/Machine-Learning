#!/usr/bin/env python
# coding=utf-8

import os
import math
import sys

tmp = [
    [17,18,24,47,99,99,99,99],
    [18,21,26,66,99,99,99,99],
    [24,26,56,99,99,99,99,99],
    [47,66,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99],
    [99,99,99,99,99,99,99,99]
]
tmp1 = [
    [16,11,10,16,24,40,51,61],
    [12,12,14,19,26,58,60,55],
    [14,13,16,24,40,57,69,56],
    [14,17,22,29,51,87,80,62],
    [18,27,37,56,68,109,103,77],
    [24,35,55,64,81,104,113,92],
    [49,64,78,87,103,121,120,101],
    [72,92,95,98,112,100,103,99]
]

def READ():
    array2 = [
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ]
    f = open('data.txt')
    line = f.readline()
    i = 0
    while line:
        l = line.split(' ')
        for m in range(0,len(l)):
            if m == 7:
                l[m] = l[m][0:2]
            array2[i][m] = int(l[m],16)
        i += 1
        line = f.readline()
    f.close()
    return array2

def WRITE(array):
    f = open('hello.txt','a')
    for i in range(0,8):
        for j in range(0,8):
            z = str(array[i][j])
            v = 5 - len(z)
            li = ' '*v + z + ' '*3
            f.write(li)
        f.write('\n')
    f.close()
def WRITEINFO(str):
    f = open('hello.txt','a')
    f.write(str)
    f.write('\n')
    f.close()


def HEX(array):
    for i in range(0,8):
        for j in range(0,8):
            array[i][j] = int(str(array[i][j]),16)

def FDCT(array):
    array2 = [
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ]


    for i in range(0,8):
        for j in range(0,8):
            if i == 0:
                A = math.sqrt(1.0/2)
            else:
                A = 1
            if j == 0:
                B = math.sqrt(1.0/2)
            else:
                B = 1
            sum = 0
            for x in range(0,8):
                for y in range(0,8):
                    sum += (array[x][y]-128)*math.cos((2*x+1)*i*math.pi/16)*math.cos((2*y+1)*j*math.pi/16)
                array2[i][j] = round(1.0/4 * A * B * sum,1) 
    return array2

def QUA(array):
    for i in range(0,8):
        for j in range(0,8):
            array[i][j] = int(round(array[i][j]/tmp[i][j],0))
    return array

def DQUA(array):
    for i in range(0,8):
        for j in range(0,8):
            array[i][j] = array[i][j]*tmp[i][j]
    return array

def IDCT(array):
    array2 = [
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ]


    for i in range(0,8):
        for j in range(0,8):
            sum = 0
            for x in range(0,8):
                for y in range(0,8):
                    if x == 0:
                        A = math.sqrt(1.0/2)
                    else:
                        A = 1
                    if y == 0:
                        B = math.sqrt(1.0/2)
                    else:
                        B = 1
                    sum += array[x][y]*math.cos((2*i+1)*x*math.pi/16)*math.cos((2*j+1)*y*math.pi/16)*A*B
                array2[i][j] = int(sum/4 + 128)
    return array2

def Print(array):
    for i in range(0,8):
        for j in range(0,8):
            z = str(array[i][j])
            v = 5 - len(z)
            sys.stdout.write(' '*v)
            sys.stdout.write(z)
            sys.stdout.write(' '*3)
        sys.stdout.flush()
        print

if __name__ == '__main__':
    a = [
        [139,144,149,153,155,155,155,155],
        [144,151,153,156,159,156,156,156],
        [150,155,160,163,158,156,156,156],
        [159,161,162,160,160,159,159,159],
        [159,160,161,162,162,155,155,155],
        [161,161,161,161,160,157,157,157],
        [162,162,161,163,162,157,157,157],
        [162,162,161,161,163,158,158,158]
    ]

    print('源图像样本：')
    WRITEINFO('源图像样本：')
    Print(READ())
    WRITE(READ())
    #print('源图像样本：')
    #Print(a)
    yy = READ()
    print('FDCT系数：')
    WRITEINFO('FDCT系数：')
    aa = FDCT(yy)
    Print(aa)
    WRITE(aa)
    print('色差量化表：')
    WRITEINFO('色差量化表：')
    Print(tmp)
    WRITE(tmp)
    print('量化系数：')
    WRITEINFO('量化系数')
    bb = QUA(aa)
    Print(bb)
    WRITE(bb)
    print('反量化系数：')
    WRITEINFO('反量化系数：')
    cc = DQUA(bb)
    Print(cc)
    WRITE(cc)
    print('重构的图像样本：')
    WRITEINFO('重构的图像样本：')
    dd = IDCT(cc)
    Print(dd)
    WRITE(dd)

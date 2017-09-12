#coding:utf-8
import json
import numpy as np
import cv2
import os
import time
import urllib


def draw(img,results):
    if results is not None:

        total_boxes = results[0]
        points = results[1]

        # for i, chip in enumerate(chips):
        #     cv2.imshow('chip_'+str(i), chip)
        #     cv2.imwrite('chip_'+str(i)+'.png', chip)

        draw = img.copy()
        #画矩形
        #定义矩形的左上顶点和右下顶点位置
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
        #画五个点
        #只需定义圆的中心点坐标和圆的半径只需定义圆的中心点坐标和圆的半径以及线的颜色和宽度
        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

        cv2.imshow("detection result", draw)
        cv2.waitKey(0)

if __name__ == '__main__':
    with open("./data.json",'r') as load_f:
        data = json.load(load_f)
        for i in range(len(data)):
            status = data[i]["status"]
            img_url = data[i]["url"]
            if status == 'none':
                print 'no face detected....url:'
                print img_url
            else:
                total_boxes = data[i]["total_boxes"]
                points = data[i]["points"]
                # chips = data[i]["chips"]
                type = data[i]["type"]

                total_boxes = np.array(total_boxes)
                points = np.array(points,dtype='float32')
                # chips = np.array(chips,dtype='uint8')

                #还原回tuple类型
                detectData = (total_boxes,points)

                if type == 'local':
                    image = cv2.imread(img_url)
                else:
                    resp = urllib.urlopen(img_url)
                    image = np.asarray(bytearray(resp.read()), dtype="uint8")
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                draw(image,detectData)

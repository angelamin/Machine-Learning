# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time



def detect(url):
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)

    print url
    img = cv2.imread(url)
    print 'hhhh'
    print img
    # run detector
    results = detector.detect_face(img)
    # print results

    if results is not None:

        total_boxes = results[0]
        points = results[1]

        # extract aligned face chips 提取对准的面部特点
        chips = detector.extract_image_chips(img, points, 144, 0.37)
        for i, chip in enumerate(chips):
            cv2.imshow('chip_'+str(i), chip)
            cv2.imwrite('chip_'+str(i)+'.png', chip)

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
    return results

# --------------
# test on camera
# --------------
'''
camera = cv2.VideoCapture(0)
while True:
    grab, frame = camera.read()
    img = cv2.resize(frame, (320,180))

    t1 = time.time()
    results = detector.detect_face(img)
    print 'time: ',time.time() - t1

    if results is None:
        continue

    total_boxes = results[0]
    points = results[1]

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
    cv2.imshow("detection result", draw)
    cv2.waitKey(30)
'''

if __name__ == "__main__":
    for filename in os.listdir(r"./imgs"):
        print r"./imgs/"+filename
        detect(r"./imgs/" + filename)

import cv2
import numpy as np
import _tkinter
from matplotlib import pyplot

img = cv2.imread('imgs/1.jpg', 0)
pyplot.imshow(img, cmap='gray', interpolation='bicubic')
pyplot.xticks([]), pyplot.yticks([])  # to hide tick values on X and Y axis
pyplot.show()

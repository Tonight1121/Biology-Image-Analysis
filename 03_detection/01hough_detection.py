import cv2
import math
import numpy as np
import os
import sys
import copy
import math as m
import os
import imageio
import matplotlib.pyplot as plt
# worth to mention, cv <col, row>, np <row, col>
pi = math.pi
# sigma = 1
import cv2
import numpy as np

import math

fName = 'red-blood-cells.png'
# fName = 'singleframe2.jpg'

if __name__ == '__main__':

    img = cv2.imread(fName, 0)
    cimg = copy.copy(img)

    I = imageio.imread(fName)
    # ILog = np.log2(I, dtype=np.float32)
    hist, bins = np.histogram(I, bins=50)
    print('hist = \n{}'.format(hist))
    print('bins = \n{}'.format(bins))
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    # plt.show()
    plt.savefig('ILog_histogram.pdf')

    ret, thresh = cv2.threshold(img, 220, 245, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite('thresh.jpg', thresh)

    edge = cv2.Canny(img, 30, 75)

    img = cv2.GaussianBlur(thresh, (5, 5), 0)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 25,
                               param1=50, param2=15,
                               minRadius=15, maxRadius=40)
    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
    #                            param1=50, param2=20,
    #                            minRadius=10, maxRadius=50)
    circles = np.uint16(np.around(circles))
    print('circles\n{}'.format(circles))

    cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2RGB)
    # cimg = cv2.imread(fName, 1)

    for i in circles[0, :]:
        ''' Draw boundary circles '''
        cv2.circle(cimg, (i[0], i[1]), i[2], (255, 0, 0), 3)
        ''' Draw the centers of the circles '''
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 5)

    # plt.imshow(cimg)
    result = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
    cv2.imwrite('results/cimg.jpg', result)
    # cv2.imshow('result', result)
    # cv2.waitKey(0)

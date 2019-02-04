import cv2
import math
import numpy as np
import os
import sys
import copy
import math as m
import os
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

    edge = cv2.Canny(img, 30, 75)

    img = cv2.GaussianBlur(img, (5, 5), 0)

    # circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 25,
    #                            param1=50, param2=15,
    #                            minRadius=15, maxRadius=26)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 25,
                               param1=50, param2=15,
                               minRadius=15, maxRadius=26)
    circles = np.uint16(np.around(circles))
    print('circles\n{}'.format(circles))

    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

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

import cv2
import math
import numpy as np
from PIL import Image
import os
import scipy.misc
import sys
import skimage
import scipy
import imageio
import matplotlib.pyplot as plt
import time
from scipy.ndimage import measurements
import copy
import seaborn as sns
from skimage.filters import threshold_otsu, threshold_local
# worth to mention, cv <col, row>, np <row, col>

fName = 'T.png'
# fName = 'letter.png'

def otsu_segmentation(img):
    # global thresholding
    ret1, th1 = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
    # plot all the images and their histograms
    # images = [img, 0, th1,
    #          img, 0, th2,
    #           blur, 0, th3]
    # titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
    #           'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
    #           'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    # plt.figure()
    # for i in range(3):
    #     plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    #     plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    #     plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    #     plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    #     plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    # plt.savefig('otsu.jpg')
    # plt.show()
    cv2.imwrite('otsu.jpg', th2)
    cv2.imwrite('otsu_orginal.jpg', th3)

def my_meanc(img, neighbour=7, C=30):
    since = time.time()

    height, width = img.shape
    # print('height {}, width {}'.format(height, width))
    margin = neighbour // 2
    # print('margin {}'.format(margin))

    space_cube = np.zeros((height-margin*2, width-margin*2, 1))
    # print('space_cube shape {}'.format(space_cube.shape))

    for move_h in range(0, neighbour):
        for move_w in range(0, neighbour):
            start_h = move_h
            start_w = move_w
            end_h = move_h - margin * 2
            end_w = move_w - margin * 2
            if move_h == margin * 2:
                end_h = None
            if move_w == margin * 2:
                end_w = None
            this_map = img[start_h:end_h, start_w:end_w]
            this_map = np.reshape(this_map, (this_map.shape[0], this_map.shape[1], 1))
            space_cube = np.concatenate((space_cube, this_map), axis=2)
            # print('({},{}) map shape {}'.format(move_h, move_w, this_map.shape))

    # print('final shape {}'.format(space_cube.shape))
    neighbour_space = space_cube[:, :, 1:]
    # print('neighbour_space shape {}'.format(neighbour_space.shape))

    center_img = img[margin:-margin, margin:-margin]
    neighbour_mean = np.mean(neighbour_space, axis=2)
    neighbour_median = np.median(neighbour_space, axis=2)
    # print('center_img shape {}'.format(center_img.shape))
    # print('neighbour_mean shape {}'.format(neighbour_mean.shape))
    # print('neighbour_median shape {}'.format(neighbour_median.shape))

    # mask_mean = (center_img <= neighbour_mean) * 0 + (center_img > neighbour_mean) * 255
    mask_mean = (center_img+C <= neighbour_mean) * 0 + (center_img+C > neighbour_mean) * 255
    # mask_median = (center_img+C <= neighbour_median) * 0 + (center_img+C > neighbour_median) * 255
    cv2.imwrite('mask_mean.jpg', mask_mean)
    # cv2.imwrite('mask_median.jpg', mask_median)
    time_elapsed = time.time() - since
    print('My_meanc completed in {}'.format(time_elapsed))

def my_medianc(img, neighbour=7, C=30):
    since = time.time()

    height, width = img.shape
    # print('height {}, width {}'.format(height, width))
    margin = neighbour // 2
    # print('margin {}'.format(margin))

    space_cube = np.zeros((height-margin*2, width-margin*2, 1))
    # print('space_cube shape {}'.format(space_cube.shape))

    for move_h in range(0, neighbour):
        for move_w in range(0, neighbour):
            start_h = move_h
            start_w = move_w
            end_h = move_h - margin * 2
            end_w = move_w - margin * 2
            if move_h == margin * 2:
                end_h = None
            if move_w == margin * 2:
                end_w = None
            this_map = img[start_h:end_h, start_w:end_w]
            this_map = np.reshape(this_map, (this_map.shape[0], this_map.shape[1], 1))
            space_cube = np.concatenate((space_cube, this_map), axis=2)
            # print('({},{}) map shape {}'.format(move_h, move_w, this_map.shape))

    # print('final shape {}'.format(space_cube.shape))
    neighbour_space = space_cube[:, :, 1:]
    # print('neighbour_space shape {}'.format(neighbour_space.shape))

    center_img = img[margin:-margin, margin:-margin]
    neighbour_mean = np.mean(neighbour_space, axis=2)
    neighbour_median = np.median(neighbour_space, axis=2)
    # print('center_img shape {}'.format(center_img.shape))
    # print('neighbour_mean shape {}'.format(neighbour_mean.shape))
    # print('neighbour_median shape {}'.format(neighbour_median.shape))

    # mask_mean = (center_img <= neighbour_mean) * 0 + (center_img > neighbour_mean) * 255
    # mask_mean = (center_img+C <= neighbour_mean) * 0 + (center_img+C > neighbour_mean) * 255
    mask_median = (center_img+C <= neighbour_median) * 0 + (center_img+C > neighbour_median) * 255
    # cv2.imwrite('mask_mean.jpg', mask_mean)
    cv2.imwrite('mask_median.jpg', mask_median)
    time_elapsed = time.time() - since
    print('My_medianc completed in {}'.format(time_elapsed))


def otsu(image):
    since = time.time()

    # global_thresh = threshold_otsu(image)
    # binary_local = (image > global_thresh) * 255 + (image <= global_thresh) * 0
    # cv2.imwrite('otsu_global.jpg', binary_local)
    # binary_global = image > global_thresh

    block_size = 35
    local_thresh = threshold_local(image, block_size, offset=10)
    binary_local = (image > local_thresh) * 255 + (image <= local_thresh) * 0
    cv2.imwrite('otsu.jpg', binary_local)

    time_elapsed = time.time() - since
    # print('Otsu complete in {:.0f}m {:.0f}s'.format(
    #     time_elapsed // 60, time_elapsed % 60))
    print('Otsu completed in {}'.format(time_elapsed))

def mean_c(img):
    since = time.time()

    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 5)
    cv2.imwrite('mean_C.jpg', th3)

    time_elapsed = time.time() - since
    print('Mean-C completed in {}'.format(time_elapsed))

if __name__ == '__main__':
    img = cv2.imread(fName, 0)

    otsu_segmentation(img)

    otsu(img)

    mean_c(img)

    my_meanc(img, neighbour=7, C=4)
    my_medianc(img, neighbour=7, C=4)


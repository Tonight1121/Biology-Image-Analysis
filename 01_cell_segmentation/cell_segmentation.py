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
# worth to mention, cv <col, row>, np <row, col>

fName = 'GFP_06-DAPI.tif'
log_threshold = 4.5
cell_size_thresh = 500

if __name__ == '__main__':
    img = cv2.imread(fName, 0)
    img = np.log2(img, dtype=np.float32)
    img = cv2.medianBlur(img, 5)
    ret, thresh = cv2.threshold(img, log_threshold, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed_copy = copy.copy(closed)
    cv2.imwrite('closed.jpg', closed)
    print('closed mask saved.')

    I = imageio.imread(fName)
    ILog = np.log2(I, dtype=np.float32)
    hist, bins = np.histogram(ILog, bins=50)
    print('hist = \n{}'.format(hist))
    print('bins = \n{}'.format(bins))
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.savefig('ILog_histogram.jpg')

    idx = ILog > log_threshold
    closed_mask = closed == 0

    '''Use idx for none-corrosion method, use closed_mask for corrosion method'''
    '''closed_mask is better, as it segmented cells more individually'''
    # ILabel, nFeatures = measurements.label(idx)
    ILabel, nFeatures = measurements.label(closed_mask)

    ''' Calculate the size for each cell and save in an array '''
    idx_size = np.zeros((nFeatures, 2), dtype=np.int)
    for i in range(0, nFeatures):
        cell_index = i + 1
        cell_size = np.count_nonzero(ILabel == cell_index)
        idx_size[i][0] = cell_index
        idx_size[i][1] = cell_size

    ''' Sort the cell from big to small based on size '''
    sort_cells = idx_size[np.argsort(-idx_size[:, 1])]

    ''' If no corrosion, then rule out those small noise according to cell_size_thresh '''
    ''' Else then no difference, already ruled out'''
    ''' Color each cell with its size rank'''
    real_cell = 0
    activation_mask = copy.copy(ILabel)
    for i in range(0, sort_cells.shape[0]):
        cell_index = sort_cells[i][0]
        cell_size = sort_cells[i][1]
        size_rank = i
        if cell_size > cell_size_thresh:
            replace_value = size_rank
            real_cell = real_cell + 1
        else:
            replace_value = 0
        activation_mask = np.where(activation_mask == cell_index, replace_value, activation_mask)

    ''' Draw heatmap according to rank of size, no need normalization '''
    ax = sns.heatmap(activation_mask, robust=True)
    fig = ax.get_figure()
    fig.savefig('heatmap.jpg')
    fig.clf()
    print('Heatmap has been saved.')

    print('There are {} cells in the image.'.format(nFeatures))
    print('The mean of size is {}.'.format(np.mean(sort_cells[:, 1])))
    print('The std. of size is {}.'.format(np.std(sort_cells[:, 1])))



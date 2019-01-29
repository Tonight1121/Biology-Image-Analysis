import cv2
import math
import numpy as np
from PIL import Image
import os
import scipy
import scipy.misc
from scipy import ndimage
import sys
import skimage
import scipy
import imageio
import matplotlib.pyplot as plt
import time
from scipy.ndimage import measurements
import copy
import seaborn as sns
import SimpleITK as sitk
# worth to mention, cv <col, row>, np <row, col>

fName = 'T.png'
# fName = 'GFP_06-DAPI.tif'
# fName = 'mean_C.jpg'
log_threshold = 4.5
cell_size_thresh = 500

def sobel_edge(img):
    print("... reading image from file: " + fName)

    # %%

    '''Convert image to array and use matplotlib '''
    itkImage = sitk.ReadImage(fName)
    print('Image has been loaded.')
    print('*' * 40)
    print(itkImage)
    numpyArray = sitk.GetArrayFromImage(itkImage)
    plt.imshow(numpyArray, cmap='gray')
    plt.show()

    # %%

    itkImage = sitk.Cast(itkImage, sitk.sitkFloat32)
    grad_filter = sitk.GradientImageFilter()
    grad = grad_filter.Execute(itkImage)
    arr = sitk.GetArrayFromImage(grad)
    grad_mag = np.sum(arr ** 2, axis=2)
    plt.subplot(1, 2, 1)
    plt.imshow(numpyArray, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(grad_mag, cmap='gray')
    plt.title('Gradient Image')
    plt.show()

    # %%

    itkImage = sitk.Cast(itkImage, sitk.sitkFloat32)
    smoothed_image = sitk.SmoothingRecursiveGaussian(itkImage, 4.0)
    # fNameOut =  'SmoothingRecursiveGaussian.tif'
    # sitk.WriteImage(sitk.Cast(smooth, sitk.sitkUInt8), fNameOut)
    plt.subplot(1, 2, 1)
    plt.imshow(numpyArray, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayFromImage(smoothed_image), cmap='gray')
    plt.title('Smoothed Image')
    plt.show()

    # Compute gradient after smoothing
    grad = grad_filter.Execute(smoothed_image)
    arr = sitk.GetArrayFromImage(grad)
    smoothed_grad_mag = np.sum(arr ** 2, axis=2)
    plt.subplot(1, 3, 1)
    plt.imshow(numpyArray, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(grad_mag, cmap='gray')
    plt.title('Gradient wo smooth')
    plt.subplot(1, 3, 3)
    plt.imshow(smoothed_grad_mag, cmap='gray')
    plt.title('Smoothed Gradient')
    plt.show()

    # %%

    sobelImage = sitk.SobelEdgeDetection(itkImage)
    # fNameOut =  'SobelEdgeDetection.tif'
    # sitk.WriteImage(sitk.Cast(sobelImage, sitk.sitkUInt8), fNameOut)
    plt.subplot(1, 2, 1)
    plt.imshow(numpyArray, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayFromImage(sobelImage), cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.show()

    # %%

    sobelImageSmooth = sitk.SobelEdgeDetection(smoothed_image)
    plt.subplot(1, 2, 1)
    plt.imshow(numpyArray, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(sitk.GetArrayFromImage(sobelImageSmooth), cmap='gray')
    plt.title('Edge Detection over Smoothed Image')
    plt.show()

def sobel(img):
    im = img.astype('int32')
    dx = ndimage.sobel(im, 0)  # horizontal derivative
    dy = ndimage.sobel(im, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    imageio.imwrite('sobel.jpg', mag)

    sobel_edge = (mag > 50)*255 + (mag<=50)*0
    imageio.imwrite('sobel_edge.jpg', sobel_edge)

def mysobel(img):
    threshold = 50
    kernel_x = np.asarray([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]])
    kernel_y = np.asarray([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])
    dx = cv2.filter2D(img, -1, kernel_x)
    dy = cv2.filter2D(img, -1, kernel_y)
    mix = np.hypot(dx, dy)  # magnitude
    mix *= 255.0 / np.max(mix)  # normalize (Q&D)
    sobel_edge = (mix > threshold)*255 + (mix <= threshold)*0

    cv2.imwrite('x_gradient.jpg', dx)
    cv2.imwrite('y_gradient.jpg', dy)
    cv2.imwrite('my_sobel_edge.jpg', sobel_edge)
    time.sleep(30)


def canny_edge(img):
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 40, 60)
    cv2.imwrite('canny_edge.jpg', edges)

if __name__ == '__main__':
    img = cv2.imread(fName, 0)

    mysobel(img)

    # sobel(img)

    sobel(img)

    canny_edge(img)


import cv2
import math
import numpy as np
import os
import sys
import copy
import math as m
import os
# import imageio
import vtk
import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import path
import mhd_utils as mu
import time
import SimpleITK as sitk
from SimpleITK import ObjectnessMeasureImageFilter
from skimage.filters import frangi
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import math

fName = 'red-blood-cells.png'
# fName = 'singleframe2.jpg'

def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

# input an image array
# normalize values to 0-255
def array_normalize(input_array):
    max_value = np.max(input_array)
    min_value = np.min(input_array)
    # print('max {}, min {}'.format(max_value, min_value))
    k = 255 / (max_value - min_value)
    min_array = np.ones_like(input_array) * min_value
    normalized = k * (input_array - min_array)
    return normalized

def save_slices(rawImage):
    rawImg = array_normalize(rawImage)

    for i in range(0, rawImg.shape[0]):
        cv2.imwrite('temp/view1/view1_{}.jpg'.format(i), rawImg[i, :, :])
        print('{}/{}: view1 finished'.format(i + 1, rawImg.shape[0]))

    for i in range(0, rawImg.shape[1]):
        cv2.imwrite('temp/view2/view2_{}.jpg'.format(i), rawImg[:, i, :])
        print('{}/{}: view2 finished'.format(i + 1, rawImg.shape[1]))

    for i in range(0, rawImg.shape[2]):
        cv2.imwrite('temp/view3/view3_{}.jpg'.format(i), rawImg[:, :, i])
        print('{}/{}: view3 finished'.format(i + 1, rawImg.shape[2]))

def visualization(volume):
    aRenderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(aRenderer)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    # The following reader is used to read a series of 2D slices(images)
    # that compose the volume.Theslicedimensions are set, and the
    #  pixel spacing.The data Endianness must also be specified.The reader
    #  uses the FilePrefix in combination with the slice number to construct
    # filenames using the format FilePrefix. % d.(In this case the FilePrefix
    # is the root name of the file.
    # v16 = volume
    v16 = vtk.vtkDICOMImageReader()
    v16.SetDataByteOrderToLittleEndian()
    v16.SetDirectoryName('temp/view1/')
    v16.SetDataSpacing(3.2, 3.2, 1.5)
    # v16.SetDataSpacing(0, 0, 0)

    # An isosurface, or contour value of 500 is known to correspond to the
    # skin of the patient.Once generated, a vtkPolyDataNormals filter is
    # used to create normals for smooth surface shading during rendering.
    skinExtractor = vtk.vtkContourFilter()
    skinExtractor.SetInputConnection(v16.GetOutputPort())
    skinExtractor.SetValue(0, 500)
    skinNormals = vtk.vtkPolyDataNormals()
    skinNormals.SetInputConnection(skinExtractor.GetOutputPort())
    skinNormals.SetFeatureAngle(60.0)
    skinMapper = vtk.vtkPolyDataMapper()
    skinMapper.SetInputConnection(skinNormals.GetOutputPort())
    skinMapper.ScalarVisibilityOff()

    skin = vtk.vtkActor()
    skin.SetMapper(skinMapper)

    outlineData = vtk.vtkOutlineFilter()
    outlineData.SetInputConnection(v16.GetOutputPort())
    mapOutline = vtk.vtkPolyDataMapper()
    mapOutline.SetInputConnection(outlineData.GetOutputPort())
    outline = vtk.vtkActor()
    outline.SetMapper(mapOutline)
    outline.GetProperty().SetColor(0, 0, 0)

    aCamera = vtk.vtkCamera()
    aCamera.SetViewUp(0, 0, -1)
    aCamera.SetPosition(0, 1, 0)
    aCamera.SetFocalPoint(0, 0, 0)
    aCamera.ComputeViewPlaneNormal()

    # Actors are added to the renderer.An initial camera view is created.
    # The Dolly() method moves the camera towards the FocalPoint,
    # thereby enlarging the image.
    aRenderer.AddActor(outline)
    aRenderer.AddActor(skin)
    aRenderer.SetActiveCamera(aCamera)
    aRenderer.ResetCamera()
    aCamera.Dolly(1.5)

    aRenderer.SetBackground(1, 1, 1)
    renWin.SetSize(640, 480)
    aRenderer.ResetCameraClippingRange()

    iren.Initialize()
    iren.Start()

    v16.Delete()
    skinExtractor.Delete()
    skinNormals.Delete()
    skinMapper.Delete()
    skin.Delete()
    outlineData.Delete()
    mapOutline.Delete()
    outline.Delete()
    aCamera.Delete()
    iren.Delete()
    renWin.Delete()
    aRenderer.Delete()

def visualization2():
    # prepare some coordinates
    x, y, z = np.indices((18, 18, 18))
    # x, y, z = np.indices((rawImg.shape[0], rawImg.shape[1], rawImg.shape[2]))

    # draw cuboids in the top left and bottom right corners, and a link between them
    cube1 = (x < 3) & (y < 3) & (z < 3)
    cube2 = (x >= 80) & (y >= 80) & (z >= 80)
    link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

    # combine the objects into a single boolean array
    voxels = cube1 | cube2 | link

    # set the colors of each object
    colors = np.empty(voxels.shape, dtype=object)
    colors[link] = 'red'
    colors[cube1] = 'blue'
    colors[cube2] = 'green'

    # and plot everything
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=colors, edgecolor='k')

    plt.show()

if __name__ == '__main__':

    # fn_mhd = path.join(mhd_folder, '{}_T0.mhd'.format(pid))
    fn_mhd = 'mra/mra.mhd'
    rawImg, header = mu.load_raw_data_with_mhd(fn_mhd)
    # visualization(rawImg)
    # visualization2()
    # time.sleep(30)
    print('max {}, min {}'.format(np.max(rawImg), np.min(rawImg)))
    print('raw {}'.format(rawImg.shape))

    view1 = cv2.imread('temp/view1_48.jpg', 0)
    # view1 = cv2.imread('temp/vessel.jpg', 0)
    view1 = cv2.imread('temp/view2_226.jpg', 0)
    # view1 = cv2.imread('temp/view3_253.jpg', 0)

    # view1 = rawImg[48, :, :]
    # view1 = array_normalize(view1)
    # view1 = cv2.medianBlur(view1, 9)
    output1 = frangi(view1, scale_range=(1, 15),
                     scale_step=0.05, beta1=100000,
                     beta2=10000,
                     black_ridges=False)
    # output1 = frangi(view1, scale_range=(1, 10), scale_step=2, beta1=0.5, beta2=15, black_ridges=False)
    # output1 = output1[47, :, :]
    cv2.imwrite('closed.jpg', array_normalize(output1))
    cv2.imwrite('view1.jpg', array_normalize(view1))
    plt.figure()
    plt.imshow(output1)
    # plt.show()
    plt.savefig('view2.pdf')
    print('finished')
    time.sleep(30)
    #
    #
    #
    # view1 = cv2.medianBlur(view1, 7)
    # hist, bins = np.histogram(view1, bins=24)
    # print(hist)
    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # plt.bar(center, hist, align='center', width=width)
    # plt.savefig('ILog_histogram.jpg')
    #
    # ret, thresh = cv2.threshold(view1, 60, 255, cv2.THRESH_BINARY)
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    # # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # # closed_copy = copy.copy(closed)
    # cv2.imwrite('closed.jpg', thresh)
    # print('closed mask saved.')
    #
    # time.sleep(30)
    #
    #
    # # save_slices(rawImg)
    #
    # # view2 = cv2.imread('temp/view1_48.jpg', 0)
    # reader = sitk.ImageFileReader()
    # reader.SetFileName('temp/view1_48.jpg')
    # # view1 = reader.Execute()
    # # print(view1)
    # view1 = sitk.ReadImage('temp/view1_48.jpg', sitk.sitkFloat32)
    # # output1 = np.zeros_like(view1)
    # filter = ObjectnessMeasureImageFilter()
    # print(dir(filter))
    # print(filter)
    # filter.SetAlpha(0.5)
    # filter.SetBeta(0.5)
    # filter.SetGamma(0)
    # # filter.SetScaleObjectnessMeasure(False)
    # # filter.SetBrightObject(True)
    # output1 = sitk.ObjectnessMeasure(image1=view1,
    #                                  alpha=0.1,
    #                                  beta=0.1,
    #                                  gamma=30,
    #                                  scaleObjectnessMeasure=True,
    #                                  objectDimension=1,
    #                                  brightObject=True)
    #
    #
    # # output1 = filter.Execute(view1)
    #
    # slice = sitk.GetArrayFromImage(output1)[:, :]
    # slice = array_normalize(slice)
    # # sitk.Show(slice)
    # # print(output1)
    # # writer = sitk.ImageFileWriter()
    # # writer.SetFileName('out.bmp')
    # # writer.Execute(output1)
    #
    # # output1 = view1.filter()
    # # cv2.imshow('out', output1)
    # # cv2.waitKey(0)
    # # print('filter {}'.format(filter))
    # cv2.imwrite('output1.jpg', slice)
    # print('image written')
    #
    # # time.sleep(30)



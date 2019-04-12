import cv2
import math
import numpy as np
import os
import sys
import copy
import math as m
import os
# import imageio
import io
import vtk
import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import path
import mhd_utils as mu
import math
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

if __name__ == '__main__':

    pi = math.pi
    resolution = 0.712891
    # img = io.imread('mra/mra.mhd', plugin='simpleitk')
    # img1 = sitk.ReadImage(ou)

    # fn_mhd = path.join(mhd_folder, '{}_T0.mhd'.format(pid))
    fn_mhd_fixed = 'data_registration/ct_fixed.mhd'
    fn_mhd_moving = 'data_registration/ct_moving.mhd'
    rawImg_f, header_f = mu.load_raw_data_with_mhd(fn_mhd_fixed)
    rawImg_m, header_m = mu.load_raw_data_with_mhd(fn_mhd_moving)

    # print('rawImg {}'.format(rawImg.shape))
    print('header {}'.format(header_f))

    # rawImg = array_normalize(rawImg)
    # plt.figure()
    # plt.imsave('fixed.jpg', rawImg_f)
    # plt.figure()
    # plt.imsave('moving.jpg', rawImg_m)

    rawImg_f = array_normalize(rawImg_f)
    rawImg_m = array_normalize(rawImg_m)
    # cv2.imwrite('results/fixed2.jpg', rawImg_f)
    # cv2.imwrite('results/moving2.jpg', rawImg_m)

    points_f = np.loadtxt('data_registration/ct_points_fixed.txt')
    points_m = np.loadtxt('data_registration/ct_points_moving.txt')
    # points_f = (points_f + np.asarray([207, 373])) * 1.4
    # points_m = (points_m + np.asarray([207, 373])) * 1.4

    # fixed_image = sitk.ReadImage('results/fixed2.jpg', sitk.sitkFloat32)
    # moving_image = sitk.ReadImage('results/moving2.jpg', sitk.sitkFloat32)

    print('shapes {} {}'.format(points_f.shape, points_m.shape))

    # rawImg_f = cv2.cvtColor(rawImg_f, cv2.COLOR_GRAY2BGR)
    # rawImg_m = cv2.cvtColor(rawImg_m, cv2.COLOR_GRAY2BGR)

    rawImg_f = cv2.imread('results/fixed2.jpg')
    rawImg_m = cv2.imread('results/moving2.jpg')
    rows, cols, ch = rawImg_f.shape
    diff_img = np.abs(rawImg_f-rawImg_m)
    cv2.imwrite('results/diff_img_svd_before.jpg', diff_img)

    ###############################################################################################
    origin_f = (-int(points_f[0][0]), -int(points_f[0][1]))
    origin_m = (-int(points_m[0][0]), -int(points_m[0][1]))

    points_f = (points_f + np.asarray([207, 373])) * 1.4
    points_m = (points_m + np.asarray([207, 373])) * 1.4

    np.savetxt('results/fixed_adjusted.txt', points_f)
    np.savetxt('results/moving_adjusted.txt', points_m)

    # u_f, s_f, vh_f = np.linalg.svd(points_f, full_matrices=True)
    # u_m, s_m, vh_m = np.linalg.svd(points_m, full_matrices=True)
    #
    # print('{} {} {}'.format(u_f.shape, s_f.shape, vh_f.shape))
    # print('{} {} {}'.format(u_m.shape, s_m.shape, vh_m.shape))
    #
    # print('vh_f {}'.format(vh_f))
    # print('vh_m {}'.format(vh_m))

    center_f = np.average(points_f, axis=0)
    center_m = np.average(points_m, axis=0)
    translation = center_f - center_m
    print('center_f {}\ncenter_m {}'.format(center_f, center_m))
    print('translation {}'.format(translation))
    mm_translation = translation / resolution
    print('mm_translation {}'.format(mm_translation))

    # points_m_recenter = points_m + translation
    points_m_recenter = points_m - center_m
    points_f_recenter = points_f - center_f
    print('recenter-m {}'.format(np.average(points_m_recenter, axis=0)))
    print('recenter-f {}'.format(np.average(points_f_recenter, axis=0)))

    # M = np.dot(np.transpose(points_f), points_m_recenter)
    M = np.dot(np.transpose(points_m_recenter), points_f)
    u, s, vh = np.linalg.svd(M, full_matrices=True)
    print('M shape {}'.format(M.shape))
    print('u {}\ns {}\nvh {}'.format(u.shape, s.shape, vh.shape))
    print('u {}\ns {}\nvh {}'.format(u, s, vh))

    # R = np.dot(vh, u)
    # R = np.transpose(vh)
    R = np.dot(np.transpose(vh), np.transpose(u))
    # R[0][1] = -R[0][1]
    # R = np.matmul(np.transpose(vh), np.transpose(u))
    print('R shape {}'.format(R.shape))
    print('R\n{}'.format(R))

    cos_value = R[0][0]
    sin_value = R[1][0]
    print('sin {}, cos {}'.format(sin_value, cos_value))
    radian1 = math.asin(sin_value)
    radian2 = math.acos(cos_value)
    print('radian1 {}, radian2 {}'.format(radian1, radian2))
    alpha = abs(360 * radian1 / (2 * pi))
    print('rotation degree: {}'.format(alpha))

    points_m_recenter = np.dot(points_m_recenter, np.transpose(R))
    points_m_recenter = points_m_recenter + center_f
    points_f_recenter = points_f_recenter + center_f
    # points_m_recenter = points_m_recenter * R

    transform_array = np.asarray([[1, 0, -center_m[0]],
                                  [0, 1, -center_m[1]]])

    dst = cv2.warpAffine(rawImg_m, transform_array, (cols*2, rows*2))

    transform_array = np.zeros((2, 3))
    transform_array[:, 0:2] = R
    transform_array[0, 2] = center_f[0]
    transform_array[1, 2] = center_f[1]

    dst = cv2.warpAffine(dst, transform_array, (cols, rows))
    cv2.imwrite('results/svd_register.jpg', dst)

    # M = cv2.getAffineTransform(points_m_recenter, points_f_recenter)
    # print('M {}'.format(M))

    # inverse = np.linalg.inv(points_m_recenter)
    # print('inverse shape {}'.format(inverse.shape))
    # T = np.dot(np.linalg.inv(points_m_recenter), points_f_recenter)
    # print('T\n{}'.format(T))


    mse = (points_f_recenter - points_m_recenter)
    mse = mse * mse
    mse = np.sum(mse, axis=1)
    mse = np.sqrt(mse)
    mse = np.sum(mse)
    print('mse {}'.format(mse))

    after = cv2.imread('moving_resampled.png')
    diff_after = np.abs(rawImg_f - after)
    cv2.imwrite('results/diff_after.jpg', diff_after)

    for i in range(0, points_f.shape[0]):
        # this_f = (-int(points_f[i][0]), -int(points_f[i][1]))
        # this_m = (-int(points_m[i][0]), -int(points_m[i][1]))
        #
        # x_f, y_f = rotate(origin_f, this_f, pi)
        # x_m, y_m = rotate(origin_m, this_m, pi)
        #
        # cv2.circle(rawImg_f, (x_f, y_f), 3, (0, 0, 255), -1)
        # cv2.circle(rawImg_m, (x_m, y_m), 3, (0, 0, 255), -1)

        cv2.circle(rawImg_f, (int(points_f_recenter[i][0]), int(points_f_recenter[i][1])), 8, (0, 255, 0), -1)
        cv2.circle(rawImg_f, (int(points_m_recenter[i][0]), int(points_m_recenter[i][1])), 5, (255, 0, 0), -1)
    cv2.imwrite('results/fixed_landmarks_recenter.jpg', rawImg_f)
    # cv2.imwrite('results/moving_landmarks.jpg', rawImg_m)
    #

    ###############################################################################################

    # for i in range(0, points_f.shape[0]):
    #     # this_f = (-int(points_f[i][0]), -int(points_f[i][1]))
    #     # this_m = (-int(points_m[i][0]), -int(points_m[i][1]))
    #     #
    #     # x_f, y_f = rotate(origin_f, this_f, pi)
    #     # x_m, y_m = rotate(origin_m, this_m, pi)
    #     #
    #     # cv2.circle(rawImg_f, (x_f, y_f), 3, (0, 0, 255), -1)
    #     # cv2.circle(rawImg_m, (x_m, y_m), 3, (0, 0, 255), -1)
    #
    #     cv2.circle(rawImg_f, (int(points_f[i][0]), int(points_f[i][1])), 5, (0, 255, 0), -1)
    #     # cv2.circle(rawImg_f, (int(points_f[i][0] + 256), int(points_f[i][1] + 400)), 3, (0, 0, 255), -1)
    #     cv2.circle(rawImg_m, (int(points_m[i][0]), int(points_m[i][1])), 5, (255, 0, 0), -1)
    # cv2.imwrite('results/fixed_landmarks.jpg', rawImg_f)
    # cv2.imwrite('results/moving_landmarks.jpg', rawImg_m)

    # %% Use the CenteredTransformInitializer to align the centers of
    #   the two volumes and set the center of rotation to the center
    #   of the fixed image.

    # fixed_image = sitk.ReadImage('fixed2.jpg', sitk.sitkFloat32)
    # moving_image = sitk.ReadImage('moving2.jpg', sitk.sitkFloat32)
    #
    # initial_transform = sitk.CenteredTransformInitializer(
    #     fixed_image,
    #     moving_image,
    #     sitk.Euler2DTransform(),
    #     sitk.CenteredTransformInitializerFilter.GEOMETRY)
    #
    # # initial_transform = sitk.Transform(2, sitk.sitkIdentity)
    #
    # moving_resampled = sitk.Resample(moving_image,
    #                                  fixed_image,
    #                                  initial_transform,
    #                                  sitk.sitkLinear,
    #                                  0.0,
    #                                  moving_image.GetPixelID())  # output pixel type
    #
    # # %% Start Image Registration
    #
    # registration_method = sitk.ImageRegistrationMethod()
    #
    # # Similarity metric settings.
    # # registration_method.SetMetricAsCorrelation()
    # # registration_method.SetMetricAsMeanSquares()
    # registration_method.SetMetricAsCorrelation()
    # # registration_method.SetMetricAsJointHistogramMutualInformation()
    #
    # registration_method.SetInterpolator(sitk.sitkLinear)
    #
    # # Optimizer settings.
    # registration_method.SetOptimizerAsGradientDescent(
    #     learningRate=0.1,
    #     numberOfIterations=1000,
    #     convergenceMinimumValue=1e-6,
    #     convergenceWindowSize=10)
    #
    # # The number of iterations involved in computations are defined by
    # # the convergence window size
    #
    # # Estimating scales of transform parameters a step sizes, from the
    # # maximum voxel shift in physical space caused by a parameter change.
    # registration_method.SetOptimizerScalesFromPhysicalShift()
    #
    # # Initialize registration
    # registration_method.SetInitialTransform(initial_transform, inPlace=False)
    #
    #
    # # %%
    #
    # # Callback invoked when the StartEvent happens, sets up our new data.
    # def clear_values():
    #     global metric_values
    #
    #     metric_values = []
    #
    #
    # # Callback invoked when the IterationEvent happens, update our data
    # # and display new figure.
    # def save_values(registration_method):
    #     global metric_values
    #     value = registration_method.GetMetricValue()
    #     metric_values.append(value)
    #     # print('It {}: metric value {:.4f}'.format(
    #     #    len(metric_values), value))
    #
    #
    # # Connect all of the observers so that we can perform plotting
    # # during registration.
    # registration_method.AddCommand(sitk.sitkStartEvent,
    #                                clear_values)
    # registration_method.AddCommand(sitk.sitkIterationEvent,
    #                                lambda: save_values(registration_method))
    #
    # final_transform = registration_method.Execute(fixed_image,
    #                                               moving_image)
    #
    # print('Final metric value: {0}'.format(
    #     registration_method.GetMetricValue()))
    # print('Optimizer\'s stopping condition, {0}'.format(
    #     registration_method.GetOptimizerStopConditionDescription()))
    #
    # moving_resampled = sitk.Resample(moving_image,
    #                                  fixed_image,
    #                                  final_transform,
    #                                  sitk.sitkNearestNeighbor,
    #                                  0.0,
    #                                  moving_image.GetPixelID())
    #
    # moving_resampled = sitk.Cast(moving_resampled, sitk.sitkUInt8)
    # sitk.WriteImage(moving_resampled, 'results/moving_resampled_interpolation_nearest.png')
    # sitk.WriteTransform(final_transform, 'results/final_transform.txt')







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
# import vtk
import matplotlib.pyplot as plt
import cv2
import numpy as np
import sklearn
import pickle
from sklearn.datasets import load_digits
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import roc_curve, auc
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import CCA
from os import path
# import mhd_utils as mu
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

def svm_classifier():

    # filename = 'train/{}_svm.pickle'.format(class_name)
    gt_vectors = np.loadtxt('data/ASD_first_83_NEU_last_76_Metabolites_Ratios.txt')
    gt_labels = np.loadtxt('data/labels.txt')
    gt_labels = np.reshape(gt_labels, (gt_labels.shape[0], 1))
    data = np.concatenate((gt_labels, gt_vectors), axis=1)
    np.random.shuffle(data)
    ratio = math.floor(data.shape[0] * 0.8)
    part_train = data[0:ratio]
    part_test = data[ratio:]
    # print('train{}, val{}'.format(part_train.shape, part_test.shape))

    train_data = part_train[:, 1:]
    train_labels = part_train[:, 0]
    test_data = part_test[:, 1:]
    test_labels = part_test[:, 0]

    # print('train data size {}, train label size {}'.format(train_data.shape, train_labels.shape))
    # print('test data size {}, test label size {}'.format(test_data.shape, test_labels.shape))

    # train_data = sklearn.preprocessing.normalize(train_data, axis=1)
    # my_svm = sklearn.svm.LinearSVC(random_state=2, tol=1e-20, max_iter=100)
    my_svm = sklearn.svm.SVC(gamma=0.001, C=100, random_state=50, probability=True)

    # my_svm = svm.SVC(kernel='linear', C=0.0001)
    my_svm.fit(train_data, train_labels)
    # with open(filename, 'wb') as f:
    #     pickle.dump(my_svm, f)
    #     print('{} SVM model saved!'.format(class_name))
    # with open(filename, 'rb') as f:
    #     clf2 = pickle.load(f)
    train_result = my_svm.predict(train_data)
    test_result = my_svm.predict(test_data)
    # print('test_result shape {}'.format(test_result.shape))
    # print('test_labels shape {}'.format(test_labels.shape))
    # print(test_result)
    # test_score = my_svm.decision_function(test_data)
    # print('train_confidence {}'.format(test_score))
    # print('train_predicts {}'.format(test_result))
    # print(test_labels)
    # print(test_result)
    train_acc = (train_labels == train_result).sum() / float(train_labels.size)
    test_acc = (test_labels == test_result).sum() / float(test_labels.size)
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('test_score {}'.format(test_score))
    # normalized_score = array_normalize(test_score)
    # print(normalized_score[0:10])
    # normalized_score = np.reshape(normalized_score, (normalized_score.shape[0], 1))
    # print('normalized shape {}'.format(normalized_score.shape))
    # print('normalized {}'.format(normalized_score[195:205]))
    # print('result {}'.format(test_result[195:205]))
    # print('test_score shape {}'.format(test_score.shape))
    return train_acc, test_acc, my_svm
    # return normalized_score
    # return test_score
    # time.sleep(30)

def cross_validation(n):
    train = []
    test = []
    models = []
    for i in range(0, n):
        train_acc, test_acc, my_svm = svm_classifier()
        train.append(train_acc)
        test.append(test_acc)
        models.append(my_svm)
        print('{}/{}: train acc {:.4f}, test acc {:.4f}'.format(i + 1, n, train_acc, test_acc))

    best_index = test.index(max(test))
    best_svm = models[best_index]
    avg_train = sum(train) / len(train)
    avg_test = sum(test) / len(test)
    avg_test = sum(test) / len(test)
    print('avg_train {:.4f}, avg_test {:.4f}'.format(avg_train, avg_test))
    print('Best from {}: train acc {:.4f}, test acc {:.4f}'.format(best_index+1, train[best_index], test[best_index]))
    with open('my_svm.pickle', 'wb') as f:
        pickle.dump(best_svm, f)
        print('Best SVM model saved!'.format())

def auc_curve(labels, score):
    print('labels {}'.format(labels.shape))
    print('score {}'.format(score.shape))
    fpr_grd, tpr_grd, _ = roc_curve(labels, score)
    epoch_auc = auc(fpr_grd, tpr_grd)
    print(fpr_grd)
    print(tpr_grd)

    plt.plot(fpr_grd, tpr_grd, label='SVM(AUC={:.4f})'.format(epoch_auc))
    # plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('roc.pdf')
    plt.show()

def svm_classifier(gt_vectors, gt_labels):

    # filename = 'train/{}_svm.pickle'.format(class_name)
    # gt_vectors = np.loadtxt('data/gt_vectors.txt')
    # gt_labels = np.loadtxt('data/gt_labels.txt')
    # gt_labels = np.reshape(gt_labels, (gt_labels.shape[0], 1))

    data = np.concatenate((gt_labels, gt_vectors), axis=1)
    # print('data shape {}'.format(data.shape))
    np.random.shuffle(data)
    ratio = math.floor(data.shape[0] * 0.8)
    part_train = data[0:ratio]
    part_test = data[ratio:]
    # print('train{}, val{}'.format(part_train.shape, part_test.shape))

    train_data = part_train[:, 1:]
    train_labels = part_train[:, 0]
    test_data = part_test[:, 1:]
    test_labels = part_test[:, 0]

    # print('{} observations: {} for training and {} for testing.'.format(data.shape[0], train_data.shape[0], test_data.shape[0]))
    # print('train data size {}, train label size {}'.format(train_data.shape, train_labels.shape))
    # print('test data size {}, test label size {}'.format(test_data.shape, test_labels.shape))

    train_data = sklearn.preprocessing.normalize(train_data, axis=1)
    my_svm = sklearn.svm.LinearSVC(random_state=2, tol=1e-10, max_iter=10)
    # my_svm = sklearn.svm.SVC(gamma=0.001, C=100, random_state=50)

    # my_svm = svm.SVC(kernel='linear', C=0.0001)
    my_svm.fit(train_data, train_labels)
    # with open(filename, 'wb') as f:
    #     pickle.dump(my_svm, f)
    #     print('{} SVM model saved!'.format(class_name))
    # with open(filename, 'rb') as f:
    #     clf2 = pickle.load(f)
    train_result = my_svm.predict(train_data)
    test_result = my_svm.predict(test_data)
    # print('test_result shape {}'.format(test_result.shape))
    # print('test_labels shape {}'.format(test_labels.shape))
    # print(test_result)
    # test_score = my_svm.decision_function(test_data)
    # print('train_confidence {}'.format(test_score))
    # print('train_predicts {}'.format(test_result))
    # print(test_labels)
    # print(test_result)
    train_acc = (train_labels == train_result).sum() / float(train_labels.size)
    test_acc = (test_labels == test_result).sum() / float(test_labels.size)
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('train acc {:.4f}, test acc {:.4f}'.format(train_acc, test_acc))
    # print('test_score {}'.format(test_score))
    # normalized_score = array_normalize(test_score)
    # print(normalized_score[0:10])
    # normalized_score = np.reshape(normalized_score, (normalized_score.shape[0], 1))
    # print('normalized shape {}'.format(normalized_score.shape))
    # print('normalized {}'.format(normalized_score[195:205]))
    # print('result {}'.format(test_result[195:205]))
    # print('test_score shape {}'.format(test_score.shape))
    return train_acc, test_acc, my_svm

def question1():
    N = 10
    a = np.asarray([0.2, 0.3, 0.5]).reshape((3, 1))
    mu, sigma = 0, 1

    X = np.random.normal(mu, sigma, (N, 3))
    E = np.random.normal(mu, 0.05, (N, 1))

    y_obs = np.dot(X, a) + E

    Xb = np.concatenate((np.ones((N, 1)), X), axis=1)

    # Xb_inv = np.linalg.inv(Xb)
    Xb_trans = np.transpose(Xb)

    theta = np.linalg.inv(np.dot(Xb_trans, Xb))
    theta = np.dot(theta, Xb_trans)
    theta = np.dot(theta, y_obs)

    print('\nXb = \n{}'.format(Xb))
    print('\ny_obs = \n{}'.format(y_obs))
    print('\ntheta = \n{}'.format(theta))

def question2(observations=10, type=1):
    N = observations
    a = np.asarray([0.2, 0.3, 0.5]).reshape((3, 1))
    mu, sigma = 0, 1

    if type == 1:
        X = np.random.normal(mu, sigma, (N, 3))
    else:
        X = np.random.normal(mu, sigma, (N, 1))
        X = np.repeat(X, 3, axis=1)
        X = 0.99 * X + 0.01 * np.random.normal(mu, sigma, (N, 3))

    E = np.random.normal(mu, 0.05, (N, 1))

    y_obs = np.dot(X, a) + E

    Xb = np.concatenate((np.ones((N, 1)), X), axis=1)

    # Xb_inv = np.linalg.inv(Xb)
    Xb_trans = np.transpose(Xb)

    theta = np.linalg.inv(np.dot(Xb_trans, Xb))
    theta = np.dot(theta, Xb_trans)
    theta = np.dot(theta, y_obs)

    print('\nXb = \n{}'.format(Xb))
    print('\ny_obs = \n{}'.format(y_obs))
    print('\ntheta = \n{}'.format(theta))
    time.sleep(30)

    theta_reshape = theta.reshape((1, 4))
    # print(theta_reshape)
    return theta_reshape


def question4(observations=10, type=1, n_components=2):
    N = observations
    a = np.asarray([0.2, 0.3, 0.5]).reshape((3, 1))
    mu, sigma = 0, 1

    if type == 1:
        X = np.random.normal(mu, sigma, (N, 3))
    else:
        X = np.random.normal(mu, sigma, (N, 1))
        X = np.repeat(X, 3, axis=1)
        X = 0.99 * X + 0.01 * np.random.normal(mu, sigma, (N, 3))

    E = np.random.normal(mu, 0.05, (N, 1))

    y_obs = np.dot(X, a) + E

    # print('X shape {}, y_obs shape {}'.format(X.shape, y_obs.shape))

    pls2 = PLSRegression(n_components=n_components)
    pls2.fit(X, y_obs)
    y_preds = pls2.predict(X)
    # print('y_obs\n{}'.format(y_obs))
    # print('y_preds\n{}'.format(y_preds))

    compare = np.concatenate((y_obs, y_preds), axis=1)
    # print('[y_obs, y_preds]=\n{}'.format(compare))

    # params = pls2.get_params(deep=True)
    params = pls2.coef_.reshape((1, 3))
    # print(params.shape)

    x_loadings = pls2.x_loadings_
    y_loadings = pls2.y_loadings_
    # print('avg\n{}'.format(np.average(x_loadings, axis=1)))
    # print(x_loadings.shape)
    # print(y_loadings)
    return params


if __name__ == '__main__':

    # generation_type = 2
    # data_sets = 100
    # results = np.zeros((data_sets, 3))
    # latent = 1
    #
    # for i in range(data_sets):
    #     theta = question4(observations=30, type=generation_type, n_components=latent)
    #     results[i, :] = theta
    #     print('{}/{}: {}'.format(i+1, data_sets, theta))
    # plt.figure()
    # plt.boxplot(results[:, :], notch=True)
    # # plt.xticks([1, 2, 3, 4], ['E', '$a_{1}$', '$a_{2}$', '$a_{3}$'])
    # plt.xticks([1, 2, 3], ['$a_{1}$'+'\n({:.4f}$\pm${:.4f})'.format(np.mean(results[:, 0]), np.std(results[:, 0])),
    #                        '$a_{2}$'+'\n({:.4f}$\pm${:.4f})'.format(np.mean(results[:, 1]), np.std(results[:, 1])),
    #                        '$a_{3}$'+'\n({:.4f}$\pm${:.4f})'.format(np.mean(results[:, 2]), np.std(results[:, 2]))])
    # plt.title('X generated using approach {}, {} latent variables'.format(generation_type, latent))
    # plt.savefig('boxplot_{}_{}v.jpg'.format(generation_type, latent))
    # plt.savefig('boxplot_{}_{}v.pdf'.format(generation_type, latent))

    ''' Question 3 starts here! '''
    generation_type = 2

    data_sets = 100
    results = np.zeros((data_sets, 4))
    np.set_printoptions(precision=4)

    for i in range(data_sets):
        theta = question2(observations=10, type=generation_type)
        results[i, :] = theta
        print('{}/{}: {}'.format(i+1, data_sets, theta))

    print(results)

    plt.figure()
    plt.boxplot(results[:, :], notch=True)
    # plt.xticks([1, 2, 3, 4], ['E', '$a_{1}$', '$a_{2}$', '$a_{3}$'])
    plt.xticks([1, 2, 3, 4], ['E\n({:.4f}$\pm${:.4f})'.format(np.mean(results[:, 0]), np.std(results[:, 0])),
                              '$a_{1}$'+'\n({:.4f}$\pm${:.4f})'.format(np.mean(results[:, 1]), np.std(results[:, 1])),
                              '$a_{2}$'+'\n({:.4f}$\pm${:.4f})'.format(np.mean(results[:, 2]), np.std(results[:, 2])),
                              '$a_{3}$'+'\n({:.4f}$\pm${:.4f})'.format(np.mean(results[:, 3]), np.std(results[:, 3]))])
    plt.title('X generated using approach {}'.format(generation_type))
    plt.savefig('boxplot_{}_lr.jpg'.format(generation_type))
    plt.savefig('boxplot_{}_lr.pdf'.format(generation_type))






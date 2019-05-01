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

    # train_data = sklearn.preprocessing.normalize(train_data, axis=1)
    # my_svm = sklearn.svm.LinearSVC(random_state=2, tol=1e-10, max_iter=10)
    my_svm = sklearn.svm.SVC(kernel='rbf', gamma=0.001, C=100, random_state=50, probability=True)

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

if __name__ == '__main__':
    # data = np.loadtxt('data.txt')
    # print(data.shape)
    #
    # male_labels = np.zeros((9, 1))
    # female_labels = np.ones((11, 1))
    # print(male_labels.shape)
    # print(female_labels.shape)
    #
    # labels = np.concatenate((male_labels, female_labels), axis=0)
    # print(labels.shape)
    # print(labels)
    #
    # resource = np.concatenate((data, labels), axis=1)
    # print(data)
    # print(resource)

    resource = np.loadtxt('data.txt')
    # train_data = sklearn.preprocessing.normalize(resource, axis=1)

    data = resource[:, 0:-1]
    print(data.shape)
    data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    np.savetxt('normalized.csv', data)
    print(resource.shape)
    # time.sleep(30)


    # vectors = data
    # vectors = data[:, [1, 2, 3]]
    # vectors = data[:, [0, 2, 3]]
    # vectors = data[:, [0, 1, 3]]
    # data =

    # vectors = data[:, [0, 1, 2]]
    vectors = data
    labels = resource[:, -1]
    labels = np.reshape(labels, (labels.shape[0], 1))
    # print(labels)
    # time.sleep(30)


    svm_iteration = 100

    train_accs = np.zeros((svm_iteration))
    test_accs = np.zeros((svm_iteration))

    # print('20 observations: 16 for training and 4 for testing.')
    train_acc, test_acc, my_svm = None, None, None
    for i in range(svm_iteration):
        train_acc, test_acc, my_svm = svm_classifier(gt_vectors=vectors, gt_labels=labels)
        train_accs[i] = train_acc
        test_accs[i] = test_acc
        print('{}/{}: train acc {:.4f}, test acc {:.4f}'.format(i+1, svm_iteration, train_acc, test_acc))
    print('*** Mean of train acc {:.4f} and test acc {:.4f} ***'.format(np.mean(train_accs), np.mean(test_accs)))

    xs = np.arange(data.shape[0]) + 1
    xs = np.reshape(xs, (xs.shape[0], 1))
    probs = my_svm.predict_proba(data)
    preds = my_svm.predict(data)
    # print('preds {}'.format(preds))
    # print('preds shape {}'.format(preds.shape))

    tn, tp, fp, fn = 0, 0, 0, 0

    for i in range(preds.shape[0]):
        t_label = labels[i]
        my_pred = preds[i]
        if t_label == 0 and my_pred == 0:
            tn = tn + 1
        elif t_label == 0 and my_pred == 1:
            fp = fp + 1
        elif t_label == 1 and my_pred == 1:
            tp = tp + 1
        else:
            fn = fn + 1
    print('tn {}, fp {}, tp {}, fn {}'.format(tn, fp, tp, fn))

    spe = tn / (tn + fp)
    sen = tp / (tp + fn)
    print('spe {}, sen {}'.format(spe, sen))



    time.sleep(30)
    xs_probs = np.concatenate((xs, probs), axis=1)

    plt.figure()
    plt.scatter(xs_probs[0:168, 0], xs_probs[0:168, 1], color='r', marker='o', alpha=0.5, label='Positives')
    plt.scatter(xs_probs[168:, 0], xs_probs[168:, 1], color='g', marker='^', alpha=0.5, label='Negatives')
    plt.axhline(y=0.5, color='b', linestyle='--')
    plt.xlabel('Case Index')
    plt.ylabel('SVM Score (Negative Probability)')
    plt.legend(loc="upper left")
    plt.savefig('probs.pdf')
    plt.savefig('probs.jpg')
    plt.show()

    time.sleep(30)

    # male = data[0:9, :]
    # female = data[9:, :]
    # print('male {}, female {}'.format(male.shape, female.shape))
    np.set_printoptions(precision=4)

    correlation = np.corrcoef(np.transpose(data))
    covariance = np.cov(np.transpose(data))
    np.savetxt('correlation.csv', correlation)
    print(correlation.shape)
    print(correlation)
    print(covariance.shape)
    print(covariance)
    # time.sleep(30)

    # asd_correlation = np.dot(np.transpose(data), data)
    # print(asd_correlation)

    w_cor, v_cor = np.linalg.eig(correlation)
    w_cov, v_cov = np.linalg.eig(covariance)
    print('*'*20)
    print(w_cor)
    # print(v_cor)
    print(w_cov)
    # print(v_cov)

    sort_values = np.sort(w_cor)[::-1]
    print('sort_values {}'.format(sort_values))
    plt.figure()
    plt.bar(range(1, len(sort_values[:]) + 1), sort_values[:])
    plt.axhline(y=1, color='r', linestyle='--')
    plt.title('Sorted Eigenvalues of PCA for Correlation Matrix')
    plt.savefig('eigenvalues.jpg')

    # time.sleep(30)

    for i in range(5):
        plt.figure()
        plt.bar(range(1, len(v_cor[:, i])+1), v_cor[:, i])
        plt.title('{}th eigenvector of correlation'.format(i))
        plt.savefig('correlation_{}.jpg'.format(i))

        # plt.figure()
        # plt.bar(range(len(v_cov[:, i])), v_cov[:, i])
        # plt.title('{}th eigenvector of covariance'.format(i))
        # plt.savefig('eigen_vectors/covariance_{}.jpg'.format(i))

    factor_components = 2
    data_factor = FactorAnalysis(n_components=factor_components, random_state=0).fit(data)
    print('neu_factor:\n{}'.format(data_factor.components_))
    data_factor = data_factor.components_

    for i in range(factor_components):
        plt.figure()
        plt.bar(range(len(data_factor[i, :])), data_factor[i, :])
        plt.title('{}th Factor of data'.format(i))
        plt.savefig('data_f{}.jpg'.format(i))

    # plt.figure()
    # plt.bar(range(len(data_factor[0, :])), data_factor[0, :])
    # plt.title('First Factor of data')
    # plt.savefig('data_f1.jpg')
    #
    # plt.figure()
    # plt.bar(range(len(data_factor[1, :])), data_factor[1, :])
    # plt.title('Second Factor of data')
    # plt.savefig('data_f2.jpg')





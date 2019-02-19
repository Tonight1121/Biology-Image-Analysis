import cv2
import math
import numpy as np
import os
import sys
import copy
import math as m
import os
import imageio
import sklearn
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import data
# worth to mention, cv <col, row>, np <row, col>
pi = math.pi
# sigma = 1
import cv2
import numpy as np
import time
import math
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

fName = 'cardiac_us.png'
# fName = 'patch.png'
# fName = 'singleframe2.jpg'

def ccmatrix():
    PATCH_SIZE = 21

    # select some patches from grassy areas of the image
    grass_locations = [(262, 377), (269, 286), (258, 461), (417, 293), (388, 484)]

    cardiac_wall = [(262, 377)]
    rv = [(269, 286)]
    lv = [(258, 461)]
    ra = [(417, 293)]
    la = [(388, 484)]

    cardiac_wall_patches = []
    rv_patches = []
    lv_patches = []
    ra_patches = []
    la_patches = []

    for loc in cardiac_wall:
        cardiac_wall_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                    loc[1]:loc[1] + PATCH_SIZE])
    for loc in rv:
        rv_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                          loc[1]:loc[1] + PATCH_SIZE])
    for loc in lv:
        lv_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                          loc[1]:loc[1] + PATCH_SIZE])
    for loc in ra:
        ra_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                          loc[1]:loc[1] + PATCH_SIZE])
    for loc in la:
        la_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                          loc[1]:loc[1] + PATCH_SIZE])

    grass_patches = []
    for loc in grass_locations:
        grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                             loc[1]:loc[1] + PATCH_SIZE])
    #
    # # select some patches from sky areas of the image
    # sky_locations = [(54, 48), (21, 233), (90, 380), (195, 330)]
    # sky_patches = []
    # for loc in sky_locations:
    #     sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
    #                        loc[1]:loc[1] + PATCH_SIZE])

    # compute some GLCM properties each patch
    xs = []
    ys = []
    for patch in (cardiac_wall_patches + rv_patches + lv_patches + ra_patches + la_patches):
        glcm = greycomatrix(patch, [5], [0], 256, symmetric=True, normed=True)
        print('glcm {}'.format(glcm.shape))
        xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
        ys.append(greycoprops(glcm, 'correlation')[0, 0])

    # create the figure
    fig = plt.figure(figsize=(8, 8))

    # display original image with locations of patches
    # ax = fig.add_subplot(3, 2, 1)
    # ax.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
    #           vmin=0, vmax=255)
    # for (y, x) in cardiac_wall:
    #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs', color='r')
    # for (y, x) in rv:
    #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs', color='b')
    # for (y, x) in lv:
    #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs', color='g')
    # for (y, x) in ra:
    #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs', color='violet')
    # for (y, x) in la:
    #     ax.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs', color='brown')
    # ax.set_xlabel('Original Image')
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.axis('image')
    # ax = fig.add_subplot(3, 2, 1)


    # plt.figure()
    # plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest',
    #           vmin=0, vmax=255)
    # for (y, x) in cardiac_wall:
    #     plt.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'gs', color='r')
    # for (y, x) in rv:
    #     plt.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs', color='b')
    # for (y, x) in lv:
    #     plt.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs', color='g')
    # for (y, x) in ra:
    #     plt.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs', color='violet')
    # for (y, x) in la:
    #     plt.plot(x + PATCH_SIZE / 2, y + PATCH_SIZE / 2, 'bs', color='brown')
    # # plt.set_xlabel('Original Image')
    # # plt.set_xticks([])
    # # plt.set_yticks([])
    # plt.savefig('image.pdf')
    # # ax.axis('image')
    #
    

    # for each patch, plot (dissimilarity, correlation)
    # ax = fig.add_subplot(3, 2, 2)
    # ax.plot(xs[:len(cardiac_wall_patches)], ys[:len(cardiac_wall_patches)], 'go', color='r',
    #         label='cardiac_wall')
    # ax.plot(xs[len(rv_patches):], ys[len(rv_patches):], 'bo', color='b',
    #         label='rv')
    # ax.plot(xs[:len(lv_patches)], ys[:len(lv_patches)], 'go', color='g',
    #         label='lv')
    # ax.plot(xs[:len(ra_patches)], ys[:len(ra_patches)], 'go', color='violet',
    #         label='ra')
    # ax.plot(xs[:len(lv_patches)], ys[:len(lv_patches)], 'go', color='brown',
    #         label='la')
    # ax.set_xlabel('GLCM Dissimilarity')
    # ax.set_ylabel('GLCM Correlation')
    # ax.legend(loc="lower right")
    # ax = fig.add_subplot(3, 2, 2)


    # plt.figure()
    # plt.scatter(xs[:len(cardiac_wall_patches)], ys[:len(cardiac_wall_patches)], color='g',
    #         label='cardiac_wall')
    # plt.scatter(xs[len(rv_patches):], ys[len(rv_patches):], color='b',
    #         label='rv')
    # plt.scatter(xs[:len(lv_patches)], ys[:len(lv_patches)], color='r',
    #         label='lv')
    # plt.scatter(xs[:len(ra_patches)], ys[:len(ra_patches)], color='violet',
    #         label='ra')
    # plt.scatter(xs[:len(lv_patches)], ys[:len(lv_patches)], color='brown',
    #         label='la')
    # # ax.set_xlabel('GLCM Dissimilarity')
    # # ax.set_ylabel('GLCM Correlation')
    # plt.legend(loc="lower right")
    # plt.savefig('points.pdf')


    # display the image patches
    for i, patch in enumerate(grass_patches):
        ax = fig.add_subplot(3, len(grass_patches), len(grass_patches) * 1 + i + 1)
        ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
                  vmin=0, vmax=255)
        ax.set_xlabel('Grass %d' % (i + 1))

    # for i, patch in enumerate(sky_patches):
    #     ax = fig.add_subplot(3, len(sky_patches), len(sky_patches) * 2 + i + 1)
    #     ax.imshow(patch, cmap=plt.cm.gray, interpolation='nearest',
    #               vmin=0, vmax=255)
    #     ax.set_xlabel('Sky %d' % (i + 1))

    # display the patches and plot
    fig.suptitle('Grey level co-occurrence matrix features', fontsize=14)
    plt.savefig('textures.pdf')
    plt.show()
    time.sleep(30)

def get_neighbour(img, neighbour=7):
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
            print('({},{}) map shape {}'.format(move_h, move_w, this_map.shape))

    # print('final shape {}'.format(space_cube.shape))
    neighbour_space = space_cube[:, :, 1:]
    print('neighbour_space shape {}'.format(neighbour_space.shape))
    return neighbour_space


def coords_matrix(height, width, xratio=None, yratio=None, scale=False):
    x_coords = np.arange(0, height, 1)
    x_coords = np.reshape(x_coords, (height, 1))
    x_coords = np.repeat(x_coords, width, axis=1)
    x_coords = np.reshape(x_coords, (height, width, 1))
    y_coords = np.arange(0, width, 1)
    y_coords = np.reshape(y_coords, (1, width))
    y_coords = np.repeat(y_coords, height, axis=0)
    y_coords = np.reshape(y_coords, (height, width, 1))

    if scale == True:
        x_coords = x_coords / xratio
        y_coords = y_coords / yratio

    coords = np.concatenate((x_coords, y_coords), axis=2)

    # print('x_coord shape {}'.format(x_coords.shape))
    # print('y_coord shape {}'.format(y_coords.shape))
    # print('coords shape {}'.format(coords.shape))

    return coords

def kmeans(neighbour=9, cluster=6):
    image = cv2.imread(fName, 0)
    height, width = image.shape

    # neighbour = 9
    # cluster = 6

    original_start = int((neighbour - 1) / 2)
    original_end = int(-(neighbour - 1) / 2)
    original_image = image[original_start:original_end, original_start:original_end]
    original_image = np.reshape(original_image, (original_image.shape[0], original_image.shape[1], 1))
    print('original shape {}'.format(original_image.shape))
    # color = tuple(np.random.randint(0, 255, 3).tolist())
    # color = np.repeat(color, 3, axis=0)
    # time.sleep(30)

    ''' neighbour_space shape is (h-4)*(w-4)*9 '''
    neighbour_space = get_neighbour(image, neighbour=neighbour)
    neighbour_avg = np.mean(neighbour_space, axis=2)
    neighbour_avg = np.reshape(neighbour_avg, (neighbour_avg.shape[0], neighbour_avg.shape[1], 1))
    neighbour_std = np.std(neighbour_space, axis=2)
    neighbour_std = np.reshape(neighbour_std, (neighbour_std.shape[0], neighbour_std.shape[1], 1))

    coords = coords_matrix(height, width, xratio=5, yratio=5, scale=True)
    # coords = coords_matrix(height, width)
    coords = coords[original_start:original_end, original_start:original_end]
    print('coords shape {}'.format(coords.shape))

    vector_mat = np.concatenate((original_image, neighbour_avg), axis=2)
    vector_mat = np.concatenate((vector_mat, neighbour_std), axis=2)
    vector_mat = np.concatenate((vector_mat, coords), axis=2)
    vectors = np.reshape(vector_mat, (vector_mat.shape[0] * vector_mat.shape[1], vector_mat.shape[2]))
    print('vector_mat shape {}'.format(vector_mat.shape))
    print('vectors shape {}'.format(vectors.shape))

    kmeans = KMeans(n_clusters=cluster, random_state=6).fit(vectors)
    result_labels = kmeans.labels_
    print('result labels shape {}'.format(result_labels.shape))
    result_labels = np.reshape(result_labels, (vector_mat.shape[0], vector_mat.shape[1]))
    result_label = np.reshape(result_labels, (vector_mat.shape[0], vector_mat.shape[1], 1))
    result_space = np.repeat(result_label, 3, axis=2)
    # print('result space shape {}'.format(result_space.shape))

    img_bgr = image[original_start:original_end, original_start:original_end]

    color_map = np.zeros((original_image.shape[0], original_image.shape[1], 3))
    for part in range(0, cluster):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        color_1 = color[0] * np.ones_like(original_image)
        color_2 = color[1] * np.ones_like(original_image)
        color_3 = color[2] * np.ones_like(original_image)
        color = np.concatenate((color_1, color_2), axis=2)
        color = np.concatenate((color, color_3), axis=2)
        segment = img_bgr * (result_labels == part)
        color_map = color_map + (result_space == part) * color
        cv2.imwrite('segment{}.jpg'.format(part), segment)
    print('segment{}.jpg written'.format(part))
    cv2.imwrite('kmeans.jpg', color_map)

def cross_validation(n):
    train = []
    test = []
    models = []
    for i in range(0, n):
        train_acc, test_acc, my_svm = svm_classifier_all()
        # train_acc, test_acc, my_svm = svm_classifier()
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
    print('Best from {}: train acc {:.4f}, test acc {:.4f}'.format(best_index, train[best_index], test[best_index]))
    with open('my_svm.pickle', 'wb') as f:
        pickle.dump(best_svm, f)
    print('Best SVM model saved!'.format())

def svm_classifer(train_vectors, train_labels):
    # train_vectors = sklearn.preprocessing.normalize(train_vectors, axis=1)
    my_svm = sklearn.svm.SVC(gamma=0.001, C=100, random_state=50)
    # my_svm = sklearn.svm.LinearSVC(random_state=2, tol=1e-20, max_iter=1)
    my_svm.fit(train_vectors, train_labels)
    train_result = my_svm.predict(train_vectors)
    train_labels = np.reshape(train_labels, (train_labels.shape[0]))
    print('train_result {}'.format(train_result.shape))
    print('train_labels {}'.format(train_labels.shape))
    ones = np.ones_like(train_labels)
    zeros = np.zeros_like(train_labels)
    corrects = (train_result == train_labels)
    corrects = (corrects == True) * ones + (corrects == False) * zeros
    print('corrects {}'.format(corrects))
    print('corrects {}'.format(corrects.shape))
    print('train_labels shape {}'.format(train_labels.shape))
    train_acc = corrects.sum() / float(train_labels.shape[0])
    print('train_acc {}'.format(train_acc))
    test_acc = train_acc
    return my_svm

def rdm_classifer(train_vectors, train_labels):
    # train_vectors = sklearn.preprocessing.normalize(train_vectors, axis=1)
    my_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    my_clf.fit(train_vectors, train_labels)
    train_result = my_clf.predict(train_vectors)
    train_labels = np.reshape(train_labels, (train_labels.shape[0]))
    print('train_result {}'.format(train_result.shape))
    print('train_labels {}'.format(train_labels.shape))
    ones = np.ones_like(train_labels)
    zeros = np.zeros_like(train_labels)
    corrects = (train_result == train_labels)
    corrects = (corrects == True) * ones + (corrects == False) * zeros
    print('corrects {}'.format(corrects))
    print('corrects {}'.format(corrects.shape))
    print('train_labels shape {}'.format(train_labels.shape))
    train_acc = corrects.sum() / float(train_labels.shape[0])
    print('train_acc {}'.format(train_acc))
    test_acc = train_acc
    return my_clf


if __name__ == '__main__':
    image = cv2.imread(fName, 0)
    height, width = image.shape
    ccmatrix()
    # kmeans()

    neighbour = 3
    original_start = int((neighbour - 1) / 2)
    original_end = int(-(neighbour - 1) / 2)

    mask_fan = cv2.imread('masks/mask_fan.png', 1)
    mask_fan = mask_fan[original_start:original_end, original_start:original_end]

    ''' Get the mask of each class and reshape '''
    mask_ra = cv2.imread('masks/mask_ra.png', 0)
    mask_ra = mask_ra[original_start:original_end, original_start:original_end]
    mask_ra = np.reshape(mask_ra, (mask_ra.shape[0] * mask_ra.shape[1], 1))
    mask_la = cv2.imread('masks/mask_la.png', 0)
    mask_la = mask_la[original_start:original_end, original_start:original_end]
    mask_la = np.reshape(mask_la, (mask_la.shape[0] * mask_la.shape[1], 1))
    mask_rv = cv2.imread('masks/mask_rv.png', 0)
    mask_rv = mask_rv[original_start:original_end, original_start:original_end]
    mask_rv = np.reshape(mask_rv, (mask_rv.shape[0] * mask_rv.shape[1], 1))
    mask_lv = cv2.imread('masks/mask_lv.png', 0)
    mask_lv = mask_lv[original_start:original_end, original_start:original_end]
    mask_lv = np.reshape(mask_lv, (mask_lv.shape[0] * mask_lv.shape[1], 1))
    mask_wall = cv2.imread('masks/mask_wall.png', 0)
    mask_wall = mask_wall[original_start:original_end, original_start:original_end]
    mask_wall = np.reshape(mask_wall, (mask_wall.shape[0] * mask_wall.shape[1], 1))

    ''' Get the original image map '''
    original_image = image[original_start:original_end, original_start:original_end]
    original_image = np.reshape(original_image, (original_image.shape[0], original_image.shape[1], 1))
    print('original shape {}'.format(original_image.shape))

    ''' neighbour_space shape is (h-4)*(w-4)*9 '''
    neighbour_space = get_neighbour(image, neighbour=neighbour)
    neighbour_avg = np.mean(neighbour_space, axis=2)
    neighbour_avg = np.reshape(neighbour_avg, (neighbour_avg.shape[0], neighbour_avg.shape[1], 1))
    neighbour_std = np.std(neighbour_space, axis=2)
    neighbour_std = np.reshape(neighbour_std, (neighbour_std.shape[0], neighbour_std.shape[1], 1))

    ''' Use the coordinates with/without scaling '''
    coords = coords_matrix(height, width, xratio=5, yratio=5, scale=True)
    # coords = coords_matrix(height, width)
    coords = coords[original_start:original_end, original_start:original_end]
    print('coords shape {}'.format(coords.shape))

    ''' Form the feature vectors for the whole image each pixel'''
    vector_mat = np.concatenate((original_image, neighbour_avg), axis=2)
    vector_mat = np.concatenate((vector_mat, neighbour_std), axis=2)
    vector_mat = np.concatenate((vector_mat, coords), axis=2)
    vectors = np.reshape(vector_mat, (vector_mat.shape[0] * vector_mat.shape[1], vector_mat.shape[2]))
    # print('vector_mat shape {}'.format(vector_mat.shape))
    # print('vectors shape {}'.format(vectors.shape))

    # print('mask_ra shape {}'.format(mask_ra.shape))
    # print('vectors shape {}'.format(vectors.shape))

    '''Filter out the feature vectors for each mask, then used for SVM training'''
    idx_ra, = np.where(mask_ra[:, 0] == 255)
    idx_la, = np.where(mask_la[:, 0] == 255)
    idx_rv, = np.where(mask_rv[:, 0] == 255)
    idx_lv, = np.where(mask_lv[:, 0] == 255)
    idx_wall, = np.where(mask_wall[:, 0] == 255)
    ra_vectors = vectors[idx_ra, :]
    la_vectors = vectors[idx_la, :]
    rv_vectors = vectors[idx_rv, :]
    lv_vectors = vectors[idx_lv, :]
    wall_vectors = vectors[idx_wall, :]
    print('ra_vectors {}'.format(ra_vectors.shape))
    print('la_vectors {}'.format(la_vectors.shape))
    print('rv_vectors {}'.format(rv_vectors.shape))
    print('lv_vectors {}'.format(lv_vectors.shape))
    print('wall_vectors {}'.format(wall_vectors.shape))
    ra_label = 0 * np.ones((ra_vectors.shape[0], 1))
    la_label = 1 * np.ones((la_vectors.shape[0], 1))
    rv_label = 2 * np.ones((rv_vectors.shape[0], 1))
    lv_label = 3 * np.ones((lv_vectors.shape[0], 1))
    wall_label = 4 * np.ones((wall_vectors.shape[0], 1))

    train_vectors = np.concatenate((ra_vectors, la_vectors), axis=0)
    train_vectors = np.concatenate((train_vectors, rv_vectors), axis=0)
    train_vectors = np.concatenate((train_vectors, lv_vectors), axis=0)
    train_vectors = np.concatenate((train_vectors, wall_vectors), axis=0)
    print('train vectors shape {}'.format(train_vectors.shape))

    train_labels = np.concatenate((ra_label, la_label), axis=0)
    train_labels = np.concatenate((train_labels, rv_label), axis=0)
    train_labels = np.concatenate((train_labels, lv_label), axis=0)
    train_labels = np.concatenate((train_labels, wall_label), axis=0)
    print('train labels shape {}'.format(train_labels.shape))

    my_svm = svm_classifer(train_vectors, train_labels)
    # my_svm = rdm_classifer(train_vectors, train_labels)
    predictions = my_svm.predict(vectors)
    predictions = np.reshape(predictions, (vector_mat.shape[0], vector_mat.shape[1], 1))
    predictions = np.repeat(predictions, 3, axis=2)
    print(predictions)

    img_bgr = image[original_start:original_end, original_start:original_end]

    colors = [(255, 255, 0),
              (255, 0, 255),
              (0, 255, 255),
              (255, 0, 0),
              (0, 255, 0)]

    color_map = np.zeros((original_image.shape[0], original_image.shape[1], 3))
    for part in range(0, 5):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        color_1 = colors[part][0] * np.ones_like(original_image)
        color_2 = colors[part][1] * np.ones_like(original_image)
        color_3 = colors[part][2] * np.ones_like(original_image)
        color = np.concatenate((color_1, color_2), axis=2)
        color = np.concatenate((color, color_3), axis=2)
        # segment = img_bgr * (result_labels == part)
        color_map = color_map + (predictions == part) * color
        # cv2.imwrite('segment{}.jpg'.format(part), segment)
    print('segment{}.jpg written'.format(part))

    with_mask = (mask_fan == 0) * 0 + (mask_fan == 255) * color_map
    cv2.imwrite('svm.jpg', with_mask)





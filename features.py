# -*- coding: utf-8 -*-
"""
ISM Project Part 1: features.py

Created on Mon Dez 3 10:49:12 2018

@author: Sabljak
"""
import numpy as np
import cv2
import os
import pandas as pd
from PIL import Image, ImageOps
from skimage import feature, img_as_ubyte, color
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


class Features:

    """Feature Extraction, Selection and Preparation"""
    def __init__(self, im_path, data_path, dev_set_path, feature_path=None, test_set=False):
        self.im_path = im_path
        self.data_path = data_path
        self.feature_path = feature_path
        self.dev_set_path = dev_set_path
        self.test_set = test_set
        self.img = []
        self.im2 = []
        self.data = []
        self.class_names = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
        self.classes = []
        self.train_classes = []
        self.dev_classes = []
        self.feature_names = []
        self.features = []
        self.train_features = []
        self.dev_features = []
        self.test_data = []
        self.test_features = []

    def load_data(self):
        # Feature Extraction of Test Data
        if self.test_set is True and self.feature_path is None:
            # get names of test images
            data = sorted([os.path.splitext(f)[0] for f in os.listdir(self.im_path) if not f.startswith('.')])
            self.data = pd.DataFrame(data, columns=['image_id'])
            # open images, equalize histogram, then save to list of numpy arrays
            for i in range(0, self.data.shape[0]):
                im = Image.open(self.im_path + self.data.image_id[i] + '.jpg')
                im.load()
                # im_hist = ImageOps.equalize(im)
                # append to list
                # TODO: rechange
                self.img.append(np.asarray(im))

            self.features = pd.DataFrame(index=range(self.data.__len__()))

        # Feature Extraction of Training Data
        else:
            self.data = pd.read_csv(self.data_path)
            self.classes = self.data.dx

            if self.feature_path is None:
                # open images, equalize histogram, then save to list of numpy arrays
                for i in range(0, self.data.shape[0]):
                    im = Image.open(self.im_path + self.data.image_id[i] + '.jpg')
                    im.load()
                    im_hist = ImageOps.equalize(im)
                    # append to list
                    self.img.append(np.asarray(im_hist))

                self.features = self.data.drop(['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization'],
                                               axis=1)

            else:
                self.features = pd.read_csv(self.feature_path)
                self.test_data = pd.DataFrame(sorted([os.path.splitext(f)[0] for f in os.listdir(self.im_path)
                                                      if not f.startswith('.')]), columns=['image_id'])
                self.test_features = pd.read_csv(self.test_set)

    """Legendre Moments"""
    def lp(self, n, x):
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return ((2*n-1)*x*self.lp(n-1, x) - (n-1)*self.lp(n-2, x)) / n

    def beta(self, img, m, n):
        return (2 * m + 1) * (2 * n + 1) / (img.shape[0] * img.shape[1])

    def xcoord(self, img, i):
        return ((2*i)/(img.shape[1]-1)) - 1

    def ycoord(self, img, j):
        return ((2*j)/(img.shape[0]-1)) - 1

    def legendre_moment(self, img, m, n):
        s = 0
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                s += self.lp(n, self.ycoord(img, y)) * self.lp(m, self.xcoord(img, x)) * img[y, x]
        return self.beta(img, m, n) * s

    # moments for each colour channel
    def legendre_moments(self):
        # initialise feature vector
        for colour in "RGB":
            for p in range(3):
                for q in range(3):
                    self.features['Moment_{0}_L{1}{2}'.format(colour, p, q)] = 'default'
        for i in range(0, self.data.shape[0]):
            im_id = self.data['image_id'][i]
            for p in range(3):
                for q in range(3):
                    m = self.legendre_moment(self.img[i], p, q)
                    self.features.loc[self.data['image_id'] == im_id, ['Moment_R_L{0}{1}'.format(p, q)]] = m[0]
                    self.features.loc[self.data['image_id'] == im_id, ['Moment_G_L{0}{1}'.format(p, q)]] = m[1]
                    self.features.loc[self.data['image_id'] == im_id, ['Moment_B_L{0}{1}'.format(p, q)]] = m[2]
                print(m)
            self.features.to_csv('features.csv', index=False)

    # moments of black-white images
    def legendre_moments_bw(self):
        for p in range(3):
            for q in range(3):
                self.features['Moment_L{0}{1}'.format(p, q)] = 'default'
        for i in range(0, self.data.shape[0]):
            im_id = self.data['image_id'][i]
            for p in range(3):
                for q in range(3):
                    self.features.loc[self.data['image_id'] == im_id, ['Moment_L{0}{1}'.format(p, q)]] = \
                        self.legendre_moment(color.rgb2grey(self.img[i]), p, q)
            # self.features.to_csv('features.csv', index=False)

    """textures"""
    def textures(self):
        self.features['contrast'] = 'default'
        self.features['dissimilarity'] = 'default'
        self.features['homogeneity'] = 'default'
        self.features['energy'] = 'default'
        self.features['correlation'] = 'default'

        for i in range(0, self.data.shape[0]):
            im_id = self.data['image_id'][i]

            grey_co_mat = feature.greycomatrix(img_as_ubyte(color.rgb2grey(self.img[i])), [2], [0], 256, symmetric=True,
                                               normed=True)

            self.features.loc[self.data['image_id'] == im_id, ['contrast']] = \
                feature.greycoprops(grey_co_mat, prop='contrast')
            self.features.loc[self.data['image_id'] == im_id, ['dissimilarity']] = \
                feature.greycoprops(grey_co_mat, prop='dissimilarity')
            self.features.loc[self.data['image_id'] == im_id, ['homogeneity']] = \
                feature.greycoprops(grey_co_mat, prop='homogeneity')
            self.features.loc[self.data['image_id'] == im_id, ['energy']] = \
                feature.greycoprops(grey_co_mat, prop='energy')
            self.features.loc[self.data['image_id'] == im_id, ['correlation']] = \
                feature.greycoprops(grey_co_mat, prop='correlation')
        self.features.to_csv('features.csv', index=False)

    """average colour"""
    def avr_colour(self):
        self.features['average_red'] = 'default'
        self.features['average_green'] = 'default'
        self.features['average_blue'] = 'default'

        for i in range(0, self.data.shape[0]):
            im_id = self.data['image_id'][i]

            # add average color information to dataframe
            average = self.img[i].mean(axis=0).mean(axis=0)

            self.features.loc[self.data['image_id'] == im_id, ['average_red']] = average[0]
            self.features.loc[self.data['image_id'] == im_id, ['average_green']] = average[1]
            self.features.loc[self.data['image_id'] == im_id, ['average_blue']] = average[2]
        self.features.to_csv('features.csv', index=False)

    """histogram of oriented gradients"""
    def hist_grad(self):
        self.features['histo_grad'] = 'default'
        hog = []
        for i in range(0, self.data.shape[0]):
            feature.hog(image, orientations=8, pixels_per_cell=(ppc, ppc), cells_per_block=(4, 4), block_norm='L2',
                visualise=True)
            hog.append(feature.hog(color.rgb2grey(self.img[i]), block_norm='L2-Hys'))

    """skin-lesion-detection opencv"""
    def skin_lesion(self):
        self.features['area'] = 'default'
        self.features['area_variance01'] = 'default'
        self.features['area_variance02'] = 'default'
        self.features['area_variance03'] = 'default'
        self.features['area_variance1'] = 'default'
        self.features['area_variance2'] = 'default'
        self.features['area_variance3'] = 'default'
        self.features['average_blue'] = 'default'
        self.features['average_blue2'] = 'default'
        self.features['average_green'] = 'default'
        self.features['average_green2'] = 'default'
        self.features['average_red'] = 'default'
        self.features['average_red2'] = 'default'
        self.features['contrast2'] = 'default'
        self.features['correlation2'] = 'default'
        self.features['dissimilarity2'] = 'default'
        self.features['energy2'] = 'default'
        self.features['homogeneity2'] = 'default'
        self.features['m01'] = 'default'
        self.features['m02'] = 'default'
        self.features['m03'] = 'default'
        self.features['m10'] = 'default'
        self.features['m11'] = 'default'
        self.features['m12'] = 'default'
        self.features['m20'] = 'default'
        self.features['m21'] = 'default'
        self.features['m30'] = 'default'
        self.features['mu02'] = 'default'
        self.features['mu03'] = 'default'
        self.features['mu11'] = 'default'
        self.features['mu12'] = 'default'
        self.features['mu20'] = 'default'
        self.features['mu21'] = 'default'
        self.features['nu03'] = 'default'
        self.features['nu12'] = 'default'
        self.features['nu20'] = 'default'
        self.features['nu21'] = 'default'
        self.features['nu30'] = 'default'
        self.features['perimeter'] = 'default'
        self.features['symmetry'] = 'default'

        for i in range(0, self.data.shape[0]):
            im_id = self.data['image_id'][i]

            hsv = cv2.cvtColor(self.img[i], cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(self.img[i], cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (17, 17), 32)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.im2.append(im2)
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) > 4:
                ellipse = cv2.fitEllipse(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                area = hsv[int(y + (0.3 * h)):int(y + (0.8 * h)), int((0.2 * w) + x):int((0.7 * w) + x)]
                ellipse_cnt = cv2.ellipse2Poly((int(ellipse[0][0]), int(ellipse[0][1])),
                                               (int(ellipse[1][0]), int(ellipse[1][1])), int(ellipse[2]), 0, 360, 1)

            # add average color information
            mask = thresh > 0
            red_sum = np.multiply(self.img[i][:, :, 0], mask).sum()
            green_sum = np.multiply(self.img[i][:, :, 1], mask).sum()
            blue_sum = np.multiply(self.img[i][:, :, 2], mask).sum()
            area_sum = mask.sum()

            # texture information
            gCoMat = feature.greycomatrix(gray, [2], [0], 256, symmetric=True, normed=True)

            # color without segmentation
            average = self.img[i].mean(axis=0).mean(axis=0)

            # add perimeter, symmetry and area variance
            perimeter = cv2.arcLength(cnt, True)
            comp = cv2.matchShapes(cnt, ellipse_cnt, 1, 0.0)
            variance = cv2.meanStdDev(area)

            # add moment information to dataframe
            moments = cv2.moments(cnt, True)

            self.features.loc[self.data['image_id'] == im_id, ['contrast2']] = feature.greycoprops(gCoMat, prop='contrast')
            self.features.loc[self.data['image_id'] == im_id, ['dissimilarity2']] = feature.greycoprops(gCoMat, prop='dissimilarity')
            self.features.loc[self.data['image_id'] == im_id, ['homogeneity2']] = feature.greycoprops(gCoMat, prop='homogeneity')
            self.features.loc[self.data['image_id'] == im_id, ['energy2']] = feature.greycoprops(gCoMat, prop='energy')
            self.features.loc[self.data['image_id'] == im_id, ['correlation2']] = feature.greycoprops(gCoMat, prop='correlation'),
            self.features.loc[self.data['image_id'] == im_id, ['average_red']] = red_sum / area_sum
            self.features.loc[self.data['image_id'] == im_id, ['average_green']] = green_sum / area_sum
            self.features.loc[self.data['image_id'] == im_id, ['average_blue']] = blue_sum / area_sum
            self.features.loc[self.data['image_id'] == im_id, ['average_red2']] = average[0]
            self.features.loc[self.data['image_id'] == im_id, ['average_green2']] = average[1]
            self.features.loc[self.data['image_id'] == im_id, ['average_blue2']] = average[2]
            self.features.loc[self.data['image_id'] == im_id, ['perimeter']] = perimeter
            self.features.loc[self.data['image_id'] == im_id, ['symmetry']] = comp
            self.features.loc[self.data['image_id'] == im_id, ['area_variance1']] = variance[1][0][0]
            self.features.loc[self.data['image_id'] == im_id, ['area_variance2']] = variance[1][1][0]
            self.features.loc[self.data['image_id'] == im_id, ['area_variance3']] = variance[1][2][0]
            self.features.loc[self.data['image_id'] == im_id, ['area_variance01']] = variance[0][0][0]
            self.features.loc[self.data['image_id'] == im_id, ['area_variance02']] = variance[0][1][0]
            self.features.loc[self.data['image_id'] == im_id, ['area_variance03']] = variance[0][2][0]
            self.features.loc[self.data['image_id'] == im_id, ['area']] = moments['m00']
            self.features.loc[self.data['image_id'] == im_id, ['m10']] = moments['m10']
            self.features.loc[self.data['image_id'] == im_id, ['m01']] = moments['m01']
            self.features.loc[self.data['image_id'] == im_id, ['m20']] = moments['m20']
            self.features.loc[self.data['image_id'] == im_id, ['m11']] = moments['m11']
            self.features.loc[self.data['image_id'] == im_id, ['m02']] = moments['m02']
            self.features.loc[self.data['image_id'] == im_id, ['m30']] = moments['m30']
            self.features.loc[self.data['image_id'] == im_id, ['m21']] = moments['m21']
            self.features.loc[self.data['image_id'] == im_id, ['m12']] = moments['m12']
            self.features.loc[self.data['image_id'] == im_id, ['m03']] = moments['m03']
            self.features.loc[self.data['image_id'] == im_id, ['mu20']] = moments['mu20']
            self.features.loc[self.data['image_id'] == im_id, ['mu11']] = moments['mu11']
            self.features.loc[self.data['image_id'] == im_id, ['mu02']] = moments['mu02']
            self.features.loc[self.data['image_id'] == im_id, ['mu03']] = moments['mu03']
            self.features.loc[self.data['image_id'] == im_id, ['mu21']] = moments['mu21']
            self.features.loc[self.data['image_id'] == im_id, ['mu12']] = moments['mu12']
            self.features.loc[self.data['image_id'] == im_id, ['mu03']] = moments['mu03']
            self.features.loc[self.data['image_id'] == im_id, ['nu20']] = moments['nu20']
            self.features.loc[self.data['image_id'] == im_id, ['nu30']] = moments['nu30']
            self.features.loc[self.data['image_id'] == im_id, ['nu21']] = moments['nu21']
            self.features.loc[self.data['image_id'] == im_id, ['nu12']] = moments['nu12']
            self.features.loc[self.data['image_id'] == im_id, ['nu03']] = moments['nu03']

        self.features.to_csv('data/features_test_set2.csv', index=False)

    """features selection"""
    def feature_selection(self):
        # select training and validation set
        dev_im_id = pd.read_csv(self.dev_set_path)
        dev_indices = self.data.image_id.isin(dev_im_id.image)[lambda x: x]
        train_indices = self.data.image_id.isin(dev_im_id.image)[lambda x: ~x]
        self.train_features = self.features.drop(dev_indices.index)
        self.train_classes = self.classes.drop(dev_indices.index)
        self.dev_features = self.features.drop(train_indices.index)
        self.dev_classes = self.classes.drop(train_indices.index)

        # drop NAN values
        """
        nan_indices = self.features.isna().any(axis=1)[lambda x: x]
        train_nan_indices = self.train_features.isna().any(axis=1)[lambda x: x]
        dev_nan_indices = self.dev_features.isna().any(axis=1)[lambda x: x]
        self.features.dropna(inplace=True)
        self.classes.drop(nan_indices.index, axis=0, inplace=True)
        self.train_features.dropna(inplace=True)
        self.train_classes.drop(train_nan_indices.index, axis=0, inplace=True)
        self.dev_features.dropna(inplace=True)
        self.dev_classes.drop(dev_nan_indices.index, axis=0, inplace=True)
        """

        # first method: select k=10 best features with chi2-test
        # problem: choose of k beforehand
        """
        chi_selector = SelectKBest(chi2, k=15).fit(features_norm, self.classes)
        chi_support = chi_selector.get_support()
        chi_feature_names = self.features.loc[:, chi_support].columns.tolist()
        chi_drop_names = self.features.loc[:, chi_support].columns.tolist()
        chi_features = chi_selector.transform(features_norm)
        print(chi_features.shape[1], 'chi-selected features', chi_feature_names)
        """

        # drop features correlated with hair
        """
        tested with augmented hair 
        im_path = 'data/HairTest/'
        data_path = 'data/HairTest/data.csv'
        feature_path = None
        dev_set_path = None
        """
        self.features.drop(['Moment_R_L02', 'Moment_R_L11', 'Moment_R_L12', 'Moment_R_L22',
                            'Moment_G_L02', 'Moment_G_L11', 'Moment_G_L12', 'Moment_G_L22',
                            'Moment_B_L02', 'Moment_B_L11', 'Moment_B_L12', 'Moment_B_L22',
                            'contrast', 'perimeter', 'area',
                            'area_variance1', 'area_variance2', 'area_variance3'], axis=1, inplace=True)

        # normalise features values
        features_norm = scale(MinMaxScaler().fit_transform(self.features))
        # other method: (L1)/L2-based feature selection with LinearSVC
        # use SVM as a sparse estimator to reduce dimensionality
        # reason: many of the estimated coefficients are zero
        # use C to control the sparsity: the smaller C the fewer features selected
        lsvc = LinearSVC(C=0.1, penalty="l2", dual=False).fit(features_norm, np.ravel(self.classes))
        svc_selector = SelectFromModel(lsvc, prefit=True)
        svc_support = svc_selector.get_support()
        svc_feature_names = self.features.loc[:, svc_support].columns.tolist()
        svc_drop_names = self.features.loc[:, ~svc_support].columns.tolist()
        svc_features = svc_selector.transform(features_norm)
        print(svc_features.shape[1], 'svc-selected features: ', svc_feature_names)

        # according to what features to use, set self variables; here: scv_features
        self.features = svc_features
        self.train_features.drop(svc_drop_names, axis=1, inplace=True)
        self.train_features = scale(MinMaxScaler().fit_transform(self.train_features))
        self.dev_features.drop(svc_drop_names, axis=1, inplace=True)
        self.dev_features = scale(MinMaxScaler().fit_transform(self.dev_features))
        self.test_features.drop(svc_drop_names, axis=1, inplace=True)
        self.test_features = scale(MinMaxScaler().fit_transform(self.test_features))
        self.feature_names = svc_feature_names

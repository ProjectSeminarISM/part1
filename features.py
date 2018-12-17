# -*- coding: utf-8 -*-
"""
ISM Project Part 1: features.py

Created on Mon Dez 3 10:49:12 2018

@author: Sabljak
"""
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from skimage import feature, img_as_ubyte, color
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


class Features:

    """Feature Extraction, Selection and Preparation"""
    def __init__(self, im_path, data_path, feature_path=None):
        self.im_path = im_path
        self.data_path = data_path
        self.feature_path = feature_path
        self.img = []
        self.data = []
        self.class_names = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']
        self.classes = []
        self.train_classes = []
        self.dev_classes = []
        self.feature_names = []
        self.features = []
        self.train_features = []
        self.dev_features = []

    def load_data(self):
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

    """Legendre Moments"""
    def lp(self, n, x):
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return ((2*n-1)*x*self.lp(n-1, x) - (n-1)*self.lp(n-2, x)) / n

    def beta(self, m, n):
        return (2 * m + 1) * (2 * n + 1) / (self.img.shape[0] * self.img.shape[1])

    def xcoord(self, i):
        return ((2*i)/(self.img.shape[1]-1)) - 1

    def ycoord(self, j):
        return ((2*j)/(self.img.shape[0]-1)) - 1

    def legendre_moment(self, m, n):
        s = 0
        for y in range(self.img.shape[0]):
            for x in range(self.img.shape[1]):
                s += self.lp(n, self.ycoord(self.img, y)) * self.lp(m, self.xcoord(self.img, x)) * self.img[y, x]
        return self.beta(self.img, m, n) * s

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
            print(self.features.loc[self.data['image_id'] == im_id, 'image_id'])
            # self.features.to_csv('features.csv', index=False)

    # moments of black-white images
    def legendre_moments_bw(self):
        for p in range(3):
            for q in range(3):
                self.features['Moment_L{0}{1}'.format(p, q)] = 'default'
        for i in range(0, self.data.shape[0]):
            im_id = self.data['image_id'][i]
            for p in range(3):
                for q in range(3):
                    self.features.loc[self.data['image_id'] == im_id, ['Moment_L{0}{1}'.format(p, q)]] = legendre_moment(color.rgb2grey(img[i]), p, q)
            print(self.features.loc[self.data['image_id'] == im_id, 'image_id'])
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

            gCoMat = feature.greycomatrix(img_as_ubyte(color.rgb2grey(self.img[i])), [2], [0], 256, symmetric=True,
                                          normed=True)

            self.features.loc[self.data['image_id'] == im_id, ['contrast']] = \
                feature.greycoprops(gCoMat, prop='contrast')
            self.features.loc[self.data['image_id'] == im_id, ['dissimilarity']] = \
                feature.greycoprops(gCoMat, prop='dissimilarity')
            self.features.loc[self.data['image_id'] == im_id, ['homogeneity']] = \
                feature.greycoprops(gCoMat, prop='homogeneity')
            self.features.loc[self.data['image_id'] == im_id, ['energy']] = \
                feature.greycoprops(gCoMat, prop='energy')
            self.features.loc[self.data['image_id'] == im_id, ['correlation']] = \
                feature.greycoprops(gCoMat, prop='correlation')

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

    """histogram of oriented gradients"""
    def histograds(self):
        hog = []
        for i in range(0, self.data.shape[0]):
            hog.append(feature.hog(color.rgb2grey(self.img[i])), block_norm='L2-Hys')

    """features selection"""
    def feature_selection(self):

        # select training and validation set
        dev_im_id = pd.read_csv('data/val_split_info.csv')
        dev_indices = self.data.image_id.isin(dev_im_id.image)[lambda x: x]
        train_indices = self.data.image_id.isin(dev_im_id.image)[lambda x: ~x]
        self.train_features = self.features.drop(dev_indices.index)
        self.train_classes = self.classes.drop(dev_indices.index)
        self.dev_features = self.features.drop(train_indices.index)
        self.dev_classes = self.classes.drop(train_indices.index)

        # drop NAN values
        nan_indices = self.features.isna().any(axis=1)[lambda x: x]
        train_nan_indices = self.train_features.isna().any(axis=1)[lambda x: x]
        dev_nan_indices = self.dev_features.isna().any(axis=1)[lambda x: x]
        self.features.dropna(inplace=True)
        self.classes.drop(nan_indices.index, axis=0, inplace=True)
        self.train_features.dropna(inplace=True)
        self.train_classes.drop(train_nan_indices.index, axis=0, inplace=True)
        self.dev_features.dropna(inplace=True)
        self.dev_classes.drop(dev_nan_indices.index, axis=0, inplace=True)

        # normalise features values
        features_norm = MinMaxScaler().fit_transform(self.features)

        # first method: select k=10 best features with chi2-test
        # problem: choose of k beforehand
        """
        chi_selector = SelectKBest(chi2, k=10).fit(features_norm, self.classes)
        chi_support = chi_selector.get_support()
        chi_feature_names = self.features.loc[:, chi_support].columns.tolist()
        chi_drop_names = self.features.loc[:, chi_support].columns.tolist()
        chi_features = chi_selector.transform(features_norm)
        print(chi_features.shape[1], 'selected features')
        print(chi_feature_names)
        """

        # other method: (L1)/L2-based feature selection with LinearSVC
        # use SVM as a sparse estimator to reduce dimensionality
        # reason: many of the estimated coefficients are zero
        # use C to control the sparsity: smaller C -> fewer features selected
        lsvc = LinearSVC(C=0.01, penalty="l2", dual=False).fit(features_norm, self.classes)
        svc_selector = SelectFromModel(lsvc, prefit=True)
        svc_support = svc_selector.get_support()
        svc_feature_names = self.features.loc[:, svc_support].columns.tolist()
        svc_drop_names = self.features.loc[:, ~svc_support].columns.tolist()
        svc_features = svc_selector.transform(features_norm)
        print(svc_features.shape[1], ' selected features')
        print(svc_feature_names)

        # according to what features to use, set self variables; here: scv_features
        self.features = svc_features
        self.train_features.drop(svc_drop_names, axis=1, inplace=True)
        self.train_features = MinMaxScaler().fit_transform(self.train_features)
        self.dev_features.drop(svc_drop_names, axis=1, inplace=True)
        self.dev_features = MinMaxScaler().fit_transform(self.dev_features)
        self.feature_names = svc_feature_names
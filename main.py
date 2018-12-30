# -*- coding: utf-8 -*-
"""
ISM Project Part 1: main.py

Created on Sun Dez 2 17:23:34 2018

@author: Sabljak
"""
from features import *
from model import *


# Training Data
# im_path = 'data/HAM10000/'  # train images
im_path = 'data/ISIC2018_Task3_Test_Input/'  # test images
data_path = 'data/HAM10000/data.csv'
feature_path = 'data/features_all_together.csv'
dev_set_path = 'data/val_split_info.csv'
test_set = 'data/features_testset_all_together.csv'

"""
# Test Features on robustness to hair
im_path = 'data/HairTest/'
data_path = 'data/HairTest/data.csv'
feature_path = None
dev_set_path = None


# Test Data
im_path = 'data/ISIC2018_Task3_Test_Input/'
data_path = None
feature_path = None
dev_set_path = None
test_set = True
"""

if __name__ == '__main__':

    f = Features(im_path, data_path, dev_set_path, feature_path, test_set)
    f.load_data()

    # feature extraction
    # f.skin_lesion()
    # f.legendre_moments()
    # f.legendre_moments_bw()
    # f.textures()
    # f.avr_colour()

    f.feature_selection()

    # model selection: 'svc', 'decision tree', 'ada-boost', 'gaussian'
    m = Model(f, 'svc')

    m.train()
    m.predict()
    m.eval()

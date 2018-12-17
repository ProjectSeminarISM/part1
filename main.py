# -*- coding: utf-8 -*-
"""
ISM Project Part 1: main.py

Created on Sun Dez 2 17:23:34 2018

@author: Sabljak
"""
from features import *
from model import *

im_path = 'data/HAM10000/'
data_path = 'data/HAM10000/data.csv'
feature_path = 'data/features_41.csv'


if __name__ == '__main__':
    f = Features(im_path, data_path, feature_path)
    f.load_data()

    # feature extraction
    # f.legendre_moments()
    # f.legendre_moments_bw()
    # f.textures()
    # f.avr_colour()

    # feature selection
    f.feature_selection()

    # model: 'svc', 'decision tree', 'ada-boost', 'gaussian'
    m = Model(f, 'svc')
    m.train()
    m.predict()
    m.eval()




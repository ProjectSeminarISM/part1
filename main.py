# -*- coding: utf-8 -*-
"""
Pytorch MNIST example

Created on Sun Dez 2 01:23:34 2018

@author: Sabljak
"""

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage import io, exposure
from matplotlib import pyplot as plt
from features import *

#############################################
# Parameters                                #
#############################################
im_path = 'data/HAM10000/'

# load images
def load_data():
    images = []
    labels = []

    # open .csv file with image names and labels
    infos = pd.read_csv(im_path + 'labels.csv').sort_values('image_id')
    # image names
    im_ids = infos.image_id
    # number of images
    im_num = im_ids.size

    # open images, equalize histogram, then save to list of numpy arrays
    for i in range(0, im_num):
        im = Image.open(im_path + im_ids[i] + '.jpg')
        im.load()

        im_hist = ImageOps.equalize(im)
        # append to list
        images.append(np.asanyarray(im_hist))

    # get labels
    labels = infos.dx

    return infos, images, labels


if __name__ == '__main__':
    info, images, labels = load_data()

    # it will take forever to calculate the moments
    # I calculate them for all images and provide them in a csv file
    legendre_moments(info, images)


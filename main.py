# -*- coding: utf-8 -*-
"""
Pytorch MNIST example

Created on Sun Dez 2 01:23:34 2018

@author: Sabljak
"""

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io, exposure
from matplotlib import pyplot as plt

#############################################
# Parameters                                #
#############################################
im_path = 'data/HAM10000/'

# load images
def load_data():
    images = []
    labels = []

    # open .csv file with image names and labels
    infos = pd.read_csv(im_path + 'labels.csv')
    # image names
    im_ids = infos.image_id
    # number of images
    im_num = im_ids.size

    # open images, then save to list of numpy arrays
    for i in range(0, 3):
        im = Image.open(im_path + im_ids[i] + '.jpg')
        im.load()
        # histogram equalisation (and array transformation)
        im_hist = exposure.equalize_hist(np.asarray(im, dtype='int32'))
        # append to list
        images.append(im_hist)

        plt.imshow(im_hist)

    # get labels
    labels = infos.dx
    
    return images, labels




if __name__ == '__main__':
    images, labels = load_data()
    x = 0

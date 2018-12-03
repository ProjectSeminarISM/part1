# -*- coding: utf-8 -*-
"""
Pytorch MNIST example

Created on Sun Dez 3 01:23:34 2018

@author: Sabljak
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import main as m

def show_image(img):
# plot image together with grid
    plt.imshow(img, cmap='gray', norm=matplotlib.colors.Normalize(vmin=0, vmax=200, clip=True))
    #plt.xticks(np.arange(img.shape[1])-0.5)
    #plt.yticks(np.arange(img.shape[0])-0.5)
    plt.grid(True, which='major')


# geometric moments
def geometric_moment(img, p, q):
    s=0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            s += x**p * y**q * img[y, x]
    return s

# central moments
def central_moment(img, p, q):
    x0 = geometric_moment(img, 1, 0) / geometric_moment(img, 0, 0)
    y0 = geometric_moment(img, 0, 1) / geometric_moment(img, 0, 0)
    s=0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            s += (x-x0)**p * (y-y0)**q * img[y, x]
    return s

# normalised moments
def normalized_moment(img, p, q):
    return central_moment(img, p, q) / (central_moment(img, 0, 0)**(1+((p+q))/2))

# hu moments
def hu_moment(img, i):
    U = lambda p,q: normalized_moment(img, p, q)
    if i == 1:
        return U(2,0) + U(0,2)
    elif i == 2:
        return (U(2,0) - U(0,2))**2 + 4*U(1,1)**2
    elif i == 3:
        return (U(3,0) - 3*U(1,2))**2 + (3*U(2,1) - U(0,3))**2
    elif i == 4:
        return (U(3,0) + U(1,2))**2 + (U(2,1) + U(0,3))**2
    elif i == 5:
        return (U(3,0) - 3*U(1,2)) * (U(3,0) + U(1,2)) * \
                        ((U(3,0) + U(1,2))**2 - 3*(U(2,1) + U(0,3))**2) + \
                        (3*U(2,1) - U(0,3)) * (U(2,1) + U(0,3)) * \
                        (3*(U(3,0) + U(1,2))**2 - (U(2,1) + U(0,3))**2)
    elif i == 6:
        return (U(2,0) - U(0,2))*((U(3,0) + U(1,2))**2 - (U(2,1) + U(0,3))**2) \
                + 4*U(1,1)*(U(3,0) + U(1,2))*(U(2,1) + U(0,3))
    elif i == 7:
        return (3*U(2,1) - U(0,3)) * (U(3,0) + U(1,2)) * \
               ((U(3,0) + U(1,2))**2 - 3*(U(2,1) + U(0,3))**2) - \
               (U(3,0) - 3*U(1,2)) * (U(2,1) + U(0,3)) * \
               (3*(U(3,0) + U(1,2))**2 - (U(2,1) + U(0,3))**2)
    else:
        return 0

# legendre moments
def lp(n, x):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return ((2*n-1)*x*lp(n-1, x) - (n-1)*lp(n-2, x)) / n

def beta(img, m, n):
    return (2 * m + 1) * (2 * n + 1) / (img.shape[0] * img.shape[1])

def xcoord(img, i):
    return ((2*i)/(img.shape[1]-1)) - 1

def ycoord(img, j):
    return ((2*j)/(img.shape[0]-1)) - 1

def legendre_moment(img, m, n):
    s = 0
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            s += lp(n, ycoord(img, y)) * lp(m, xcoord(img, x)) * img[y, x]
    return beta(img, m, n) * s




if __name__ == '__main__':
    images, labels = m.load_data()
    img = images[0]
    show_image(img)

    for idx, img in enumerate([img]):
        print(f'Image {idx + 1}:')
        for p in range(3):
            for q in range(3):
                m = legendre_moment(img, p, q)
                print(f'L_{p}{q} = {m}')
        print()

    x = 0
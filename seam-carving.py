import sys

import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve

from tqdm import trange

def energy_calculation(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])

    filter_du = np.stack([filter_du] * 3, axis = 2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])

    filter_dv = np.stack([filter_dv] * 3, axis = 2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    energy_map = convolved.sum(axis = 2)

    print(energy_map.shape)
    energy_img = imwrite("energy_map.jpg", energy_map)
    return energy_map

img = imread("img.jpg")
energy_calculation(img)
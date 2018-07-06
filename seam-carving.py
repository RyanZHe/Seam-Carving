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

    # print(energy_map.shape)
    # energy_img = imwrite("energy_map.jpg", energy_map)
    return energy_map

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = energy_calculation(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype = np.int)

    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j -1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)

    mask = np.ones((r, c), dtype = np.bool)

    j = np.argmin(M[-1])

    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis = 2)

    img = img[mask].reshape((r, c - 1, 3))

    return img

def crop(img, scale):
    r, c, _ = img.shape
    new_c = int(scale * c)

    for i in trange(c - new_c):
        img = carve_column(img)

    return img

def main():
    scale = float(sys.argv[1])
    in_filename = sys.argv[2]
    out_filename = sys.argv[3]

    img = imread(in_filename)
    out = crop(img, scale)
    imwrite(out_filename, out)

if __name__ == "__main__":
    main()
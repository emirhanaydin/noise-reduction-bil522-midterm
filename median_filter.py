import cv2
import numpy as np


def adaptive_median_filter(img, kernel_size, noise_coordinates: set):
    pad_size = int((kernel_size - 1) / 2)
    pad_img = cv2.copyMakeBorder(img, *[pad_size] * 4, cv2.BORDER_DEFAULT)

    median = np.zeros_like(img)

    row, col = img.shape

    for v in range(row):
        for u in range(col):
            if not (v, u) in noise_coordinates:
                median[v, u] = img[v, u]
                continue
            m1 = pad_img[v: v + kernel_size, u: u + kernel_size]
            m2 = m1.reshape(-1)
            m3 = np.sort(m2)
            median[v, u] = m3[kernel_size + 1]

    return median

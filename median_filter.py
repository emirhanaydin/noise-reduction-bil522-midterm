import cv2
import numpy as np


def selective_median_filter(img, kernel_size, noise_matrix):
    pad_size = int((kernel_size - 1) / 2)
    pad_img = cv2.copyMakeBorder(img, *[pad_size] * 4, cv2.BORDER_DEFAULT)

    result = np.zeros_like(img)

    h, w = img.shape
    for v in range(h):
        for u in range(w):
            if noise_matrix[v, u][3] == 0:
                result[v, u] = img[v, u]
                continue
            m1 = pad_img[v: v + kernel_size, u: u + kernel_size]
            m2 = m1.reshape(-1)
            m3 = np.sort(m2)
            result[v, u] = m3[kernel_size + 1]

    return result

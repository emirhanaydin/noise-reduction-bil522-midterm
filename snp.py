import cv2
import numpy as np


def get_noise_matrix(img, kernel_size=3, deviation_tolerance=0.6, threshold=(127, 128)):
    pad_size = int((kernel_size - 1) / 2)
    pad_img = cv2.copyMakeBorder(img, *[pad_size] * 4, cv2.BORDER_DEFAULT)

    result = np.zeros_like(img)

    row, col = img.shape

    low, high = threshold

    for v in range(row):
        for u in range(col):
            kernel = pad_img[v: v + kernel_size, u: u + kernel_size].reshape(-1)
            mean, stddev = cv2.meanStdDev(kernel)
            el = img[v, u]
            if abs(el - mean) * deviation_tolerance < stddev:
                result[v, u] = 127
                continue

            result[v, u] = 0 if el <= low else 255 if el > high else 127

    return result


def get_noise_coordinates(noise_matrix) -> set:
    result = set()

    row, col = noise_matrix.shape

    for v in range(row):
        for u in range(col):
            el = noise_matrix[v, u]
            if el != 127:
                result.add((v, u))

    return result

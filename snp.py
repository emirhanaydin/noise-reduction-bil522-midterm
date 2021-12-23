import cv2
import numpy as np

from median_filter import selective_median_filter


def read_noise_matrix():
    img = cv2.imread("images/snp_noise.png", cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    return img


def scan_noise_matrix(img, kernel_size=3, deviation_tolerance=0.5, threshold=(127, 128)):
    pad_size = int((kernel_size - 1) / 2)
    pad_img = cv2.copyMakeBorder(img, *[pad_size] * 4, cv2.BORDER_DEFAULT)

    h, w = img.shape
    result = np.zeros([h, w, 4], np.uint8)
    low, high = threshold
    for v in range(h):
        for u in range(w):
            k = pad_img[v: v + kernel_size, u: u + kernel_size].reshape(-1)
            mean, stddev = cv2.meanStdDev(k)
            p = img[v, u]
            if abs(p - mean) * deviation_tolerance < stddev:
                continue

            result[v, u] = [0, 0, 0, 255] if p <= low else 255 if p > high else 0

    return result


def sp_denoise(img):
    sp_noise_matrix = read_noise_matrix()
    if sp_noise_matrix is None:
        sp_noise_matrix = scan_noise_matrix(img, 3, 0.74)

    return selective_median_filter(img, 3, sp_noise_matrix)

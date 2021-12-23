import cv2
import numpy as np

from magnitude_spectrum import get_magnitude_spectrum, reverse_transform


def read_reject_filter():
    img = cv2.imread("images/notch-reject-filter.png", cv2.IMREAD_UNCHANGED)

    h, w, _ = img.shape
    result = np.ones((h, w), dtype=np.uint8)

    for v in range(h):
        for u in range(w):
            if img[v, u][3] != 0:
                result[v, u] = 0

    return result


def notch_filter(img):
    reject_filter = read_reject_filter()

    f, fshift, _ = get_magnitude_spectrum(img)

    f_filtered = np.multiply(fshift, reject_filter)
    img_back = reverse_transform(f_filtered)

    return np.uint8(img_back)

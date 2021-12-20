import sys
from pathlib import Path

import cv2 as cv
import numpy as np

from median_filter import adaptive_median_filter
from snp import get_noise_matrix, get_noise_coordinates


def print_fatal(s: str):
    sys.stderr.write(s)
    sys.exit(1)


def validate_version():
    major, minor = sys.version_info[0:2]
    if major < 3 or minor < 10:
        print_fatal(f"Python interpreter must be {major}.{minor} or greater.")


def validate_path(path: Path):
    if not path.is_file():
        print_fatal("Specified file path is invalid.")


def get_input_path() -> Path:
    argv = sys.argv
    if len(argv) < 2:
        print_fatal(f"Input file path must be specified as command line argument.")

    input_path = Path(argv[1])
    validate_path(input_path)
    return input_path


def get_color_coords(img, colors):
    row, col = img.shape

    coords = set()

    snp = np.zeros_like(img)

    for v in range(row):
        for u in range(col):
            if img[v, u] in colors:
                coords.add((u, v))

            snp[v, u] = img[v, u] if img[v, u] in colors else 127

    cv.imwrite("snp.jpg", img=snp)

    return coords


def get_input_image():
    input_path = get_input_path()
    return cv.imread(str(input_path), 0)


def magnitude_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return 20 * np.log(np.abs(fshift))


def main():
    validate_version()
    img = get_input_image()

    sp_noise_matrix = get_noise_matrix(img, 3, 0.74, (20, 210))
    cv.imwrite(filename="sp_noise.jpg", img=sp_noise_matrix)

    coords = get_noise_coordinates(noise_matrix=sp_noise_matrix)
    denoise_img = adaptive_median_filter(img, 3, coords)

    cv.imwrite(filename="denoise.jpg", img=denoise_img)


if __name__ == '__main__':
    main()

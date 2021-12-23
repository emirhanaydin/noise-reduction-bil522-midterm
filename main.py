import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from mean_square_error import mean_square_error
from notch_filter import notch_filter
from snp import sp_denoise
from unsharp_mask import unsharp_mask


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


def read_image(path: str):
    path = Path(path)
    validate_path(path)
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def read_input_image():
    return read_image("images/input.png")


def read_original_image():
    return read_image("images/original.png")


def main():
    validate_version()
    in_img = read_input_image()
    orig_img = read_original_image()

    step1_img = notch_filter(in_img)
    step2_img = sp_denoise(step1_img)
    step3_img = unsharp_mask(step2_img, sigma=11, kernel_size=(11, 11))

    plots = [
        [in_img, "Input Image"],
        [step1_img, "Periodic Noise Filter"],
        [step2_img, "S&P Noise Filter"],
        [step3_img, "Sharpened - Output Image"],
    ]

    plt.rcParams["figure.figsize"] = (20, 10)
    _, axs = plt.subplots(ncols=len(plots))
    axs = axs.flatten()

    for p, ax in zip(plots, axs):
        im, title = p
        ax = ax
        ax.set_title(title)
        ax.imshow(im, cmap="gray")

    plt.show()

    print(mean_square_error(orig_img, in_img))
    print(mean_square_error(orig_img, step1_img))
    print(mean_square_error(orig_img, step2_img))
    print(mean_square_error(orig_img, step3_img))


if __name__ == '__main__':
    main()

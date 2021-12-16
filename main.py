import sys
from pathlib import Path

import cv2 as cv


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


def get_input_image():
    input_path = get_input_path()
    return cv.imread(str(input_path))


def main():
    validate_version()
    img = get_input_image()

    cv.imshow("input", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

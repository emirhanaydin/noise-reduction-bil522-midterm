import sys


def validate_version():
    major, minor = sys.version_info[0:2]
    if major < 3 or minor < 10:
        sys.stderr.write(f"Python interpreter must be {major}.{minor} or greater.")


if __name__ == '__main__':
    validate_version()

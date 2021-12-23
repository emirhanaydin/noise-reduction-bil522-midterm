import numpy as np


def get_magnitude_spectrum(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return f, fshift, magnitude_spectrum


def reverse_transform(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img = np.fft.ifft2(f_ishift)
    return np.abs(img)

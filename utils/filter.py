import numpy as np
import scipy.signal as signal
import scipy.ndimage as ndimage


def butter_highpass(signals, cutoff, fs, order=6, axis=-1):
    """ Forward-backward filter inpute signals with butterworth kernel"""
    return signal.filtfilt(*signal.butter(order,
                                          cutoff / (0.5 * fs),
                                          btype='high',
                                          analog=False),
                           signals,
                           axis=axis)


def gaussian_bandpass(signals, kernel_len, kernel_sigma, axis=-1):
    """ Convolve inputs signals with 0 mean gaussian kernel"""
    return ndimage.convolve1d(signals,
                              gaussian_kernel(kernel_len,
                                              kernel_sigma),
                              axis=axis)


def gaussian_kernel(length, sigma):
    """
    Returns a 1D gaussian filter of specified length and stdv for convolution
    """
    n = (length - 1.) / 2.
    kernel = np.exp((-1 * np.power(np.arange(-n, n + 1), 2)) / (2. * sigma**2))
    kernel[kernel < np.finfo(kernel.dtype).eps * kernel.max()] = 0
    sumh = kernel.sum()
    if sumh != 0:
        kernel /= sumh
    return kernel - kernel.mean()


def detrend(signals, degree=2):
    """ Substracts the order k trend from each input pixel """
    _, len_signal = signals.shape
    X = np.array([np.power(np.arange(len_signal), k)
                  for k in range(degree + 1)]).T
    Hat = X.dot(np.linalg.inv(X.T.dot(X))).dot(X.T)
    return signals - signals.dot(Hat)

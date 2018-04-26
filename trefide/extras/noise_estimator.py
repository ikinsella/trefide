import numpy as np
import scipy as sp

# use get_noise_fft


def noise_estimator(Y,range_ff=[0.25,0.5],method='logmexp'):
    dims = Y.shape
    if len(dims)>2:
        V_hat = Y.reshape((np.prod(dims[:2]),dims[2]),order='F')
    else:
        V_hat = Y.copy()
    sns = []
    for i in range(V_hat.shape[0]):
        ff, Pxx = sp.signal.welch(V_hat[i,:],nperseg=min(256,dims[-1]))
        ind1 = ff > range_ff[0]
        ind2 = ff < range_ff[1]
        ind = np.logical_and(ind1, ind2)
        #Pls.append(Pxx)
        #ffs.append(ff)
        Pxx_ind = Pxx[ind]
        sn = {
            'mean': lambda Pxx_ind: np.sqrt(np.mean(np.divide(Pxx_ind, 2))),
            'median': lambda Pxx_ind: np.sqrt(np.median(np.divide(Pxx_ind, 2))),
            'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(np.divide(Pxx_ind, 2)))))
        }[method](Pxx_ind)
        sns.append(sn)
    sns = np.asarray(sns)
    if len(dims)>2:
        sns = sns.reshape(dims[:2],order='F')
    return sns

def fft_estimator(signal, freq_range=[0.25, 0.5], max_samples=3072):
    """
    High frequency components of FFT of the input signal
    ________
    Input:
        signals: (len_signal,) np.ndarray
            Noise contaminated temporal signal
            (required)
        max_samples: positive integer
            Maximum number of samples which will be used in computing the
            power spectrum in the 'fft' noise estimator
            (default: 3072)
        freq_range: (2,) np.ndarray or len 2 list of increasing elements
                    between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power
            spectrum is averaged in the 'pwelch' and 'fft' noise estimators
            (default: [0.25,0.5])
    ________
    Output:
        PSD[freq_range]: np.ndarray
            Components of PSD corresponding to freq_range
    """

    # Subsample signal if length > max_samples
    len_signal = len(signal)
    if len_signal > max_samples:
        signal = np.concatenate(
            (signal[1:np.int(np.divide(max_samples, 3)) + 1],
             signal[np.int(np.divide(len_signal, 2) - max_samples / 3 / 2):
                    np.int(np.divide(len_signal, 2) + max_samples / 3 / 2)],
             signal[-np.int(np.divide(max_samples, 3)):]),
            axis=-1)
        len_signal = len(signal)

    # Create a map of freq_range on fft space
    ff = np.arange(0, 0.5 + np.divide(1., len_signal),
                   np.divide(1., len_signal))
    idx = np.logical_and(ff > freq_range[0], ff <= freq_range[1])

    # we compute the mean of the noise spectral density s
    xdft = np.fliplr(np.fft.rfft(signal))
    psdx = (np.divide(1., len_signal)) * (xdft**2)
    psdx[1:] *= 2
    return np.divide(psdx[idx[:psdx.shape[0]]], 2)

def pwelch_estimator(signal, freq_range=[0.25, 0.5]):
    """
    High frequency components of Welch's PSD estimate of the input signal
    ________
    Input:
        signals: (len_signal,) np.ndarray
            Noise contaminated temporal signal
            (required)
        freq_range: (2,) np.ndarray or len 2 list of increasing elements
                    between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power
            spectrum is averaged in the 'pwelch' and 'fft' noise estimators
            (default: [0.25,0.5])
    ________
    Output:
        PSD[freq_range]: np.ndarray
            Components of PSD corresponding to freq_range
    """
    ff, Pxx = sp.signal.welch(signal, nperseg=min(256, len(signal)))
    idx = np.logical_and(ff > freq_range[0], ff <= freq_range[1])
    return np.divide(Pxx[idx], 2)


def boot_estimator(signal, num_samples=1000, len_samples=25):
    """
    Generate bootstrapped estimated of the noise variance as the MSE of
    linear fits to small (random) subsamples of the original signal
    ________
    Input:
        signals: (len_signal,) np.ndarray
            Noise contaminated temporal signal
            (required)
        num_samples: positive integer
            Number of bootstrap MSE estimates to average over
            (default: 1000)
        len_samples: positive integer < len_signals
            Length of subsamples used in bootstrap estimates
           (default: 25)
    ________
    Output:
        mses: len num_samples list
            MSE of bootstrapped linear fits
    """

    # Precompute hat matrix to quickly generate linear predictions
    X = np.array([np.arange(len_samples), np.ones(len_samples)]).T
    Hat = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)

    # Compute mean square error of linear fit to each subsample
    return [np.mean(np.power(signal[sdx:sdx + len_samples] -
                             np.dot(Hat, signal[sdx:sdx + len_samples]), 2))
            for sdx in np.random.randint(0, len(signal) - len_samples + 1,
                                         size=num_samples)]


def estimate_noise(signals,
                   estimator='pwelch',
                   summarize='logmexp',
                   freq_range=[0.25, 0.5],
                   max_samples_fft=3072,
                   num_samples_boot=1000,
                   len_samples_boot=25):
    """
    Estimate the standard deviation of the noise contaminating temporal signals
    ________
    Input:
        signals: (num_signals, len_signals) np.ndarray or len num_signals list
                 of (len_signals,) np.ndarrays
            Collection of (gaussian) noise contaminated temporal signals (required)
        estimator: string
            Method of estimating the noise level
            Choices:
                'pwelch': average over high frequency components of Welch's
                          PSD estimate (default)
                'fft': average over high frequency components of the FFT
                'boot': bootstrap estimates of the mse of linear fits to small
                        subsamples of the signal (only appropriate when signal
                        is approximately piecewise linear)
        summarize: string
            Method of averaging the power spectrum/bootstrap samples.
            Choices:
                'mean': Mean
                'median': Median
                'logmexp': Exponential of the mean of the logs
            (default: 'logmexp')
        freq_range: (2,) np.ndarray or len 2 list of increasing elements
                    between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power
            spectrum is averaged in the 'pwelch' and 'fft' noise estimators
            (default: [0.25,0.5])
        max_samples_fft: positive integer
            Maximum number of samples which will be used in computing the
            power spectrum in the 'fft' noise estimator
            (default: 3072)
        num_samples_boot: positive integer
            Number of bootstrapped estimates of MSE to average over in the
            'boot' estimator
            (default: 1000)
        len_samples_boot: positive integer < len_signals
            Length of subsampled signals from which MSE estimated are
            generated in the 'boot' estimator
            (default: 25)
    ________
    Output:
        stdvs: (num_signals,) np.ndarray
            Estimated standard deviation for each input signal
    """
    # Assign function to summarize spectral components / bootstrap samples
    summarizer = {
        'mean': np.mean,
        'median': np.median,
        'logmexp': lambda x: np.exp(np.mean(np.log(x + 1e-11)))
    }[summarize]

    # Assign function to generate estimates of signal noise variance
    estimator = {
        'fft': lambda x: fft_estimator(x,
                                        freq_range=freq_range,
                                        max_samples=max_samples_fft),
        'pwelch': lambda x: pwelch_estimator(x,
                                              freq_range=freq_range),
        'boot': lambda x: boot_estimator(x,
                                          num_samples=num_samples_boot,
                                          len_samples=len_samples_boot)
    }[estimator]

    # Compute & return estimate of standard deviations for each signal
    return np.sqrt([summarizer(estimator(signal)) for signal in signals])


def get_noise_fft(Y, noise_range=[0.25, 0.5], noise_method='logmexp', max_num_samples_fft=3072,
                  opencv=True):
    """Estimate the noise level for each pixel by averaging the power spectral density.

    Inputs:
    -------

    Y: np.ndarray

    Input movie data with time in the last axis

    noise_range: np.ndarray [2 x 1] between 0 and 0.5
        Range of frequencies compared to Nyquist rate over which the power spectrum is averaged
        default: [0.25,0.5]

    noise method: string
        method of averaging the noise.
        Choices:
            'mean': Mean
            'median': Median
            'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Output:
    ------
    sn: np.ndarray
        Noise level for each pixel
    """
    if Y.ndim ==1:
        Y = Y[np.newaxis,:]
    #
    T = Y.shape[-1]
    # Y=np.array(Y,dtype=np.float64)

    if T > max_num_samples_fft:
        Y = np.concatenate((Y[..., 1:max_num_samples_fft // 3 + 1],
                            Y[..., np.int(T // 2 - max_num_samples_fft / 3 / 2):np.int(T // 2 + max_num_samples_fft / 3 / 2)],
                            Y[..., -max_num_samples_fft // 3:]), axis=-1)
        T = np.shape(Y)[-1]

    # we create a map of what is the noise on the FFT space
    ff = np.arange(0, 0.5 + 1. / T, 1. / T)
    ind1 = ff > noise_range[0]
    ind2 = ff <= noise_range[1]
    ind = np.logical_and(ind1, ind2)
    # we compute the mean of the noise spectral density s
    if Y.ndim > 1:
        if opencv:
            import cv2
            psdx = []
            for y in Y.reshape(-1, T):
                dft = cv2.dft(y, flags=cv2.DFT_COMPLEX_OUTPUT).squeeze()[
                    :len(ind)][ind]
                psdx.append(np.sum(1. / T * dft * dft, 1))
            psdx = np.reshape(psdx, Y.shape[:-1] + (-1,))
        else:
            xdft = np.fft.rfft(Y, axis=-1)
            xdft = xdft[..., ind[:xdft.shape[-1]]]
            psdx = 1. / T * abs(xdft)**2
        psdx *= 2
        sn = mean_psd(psdx, method=noise_method)

    else:
        xdft = np.fliplr(np.fft.rfft(Y))
        psdx = 1. / T * (xdft**2)
        psdx[1:] *= 2
        sn = mean_psd(psdx[ind[:psdx.shape[0]]], method=noise_method)

    return sn, psdx

def mean_psd(y, method='logmexp'):
    """
    Averaging the PSD

    Parameters:
    ----------

        y: np.ndarray
             PSD values

        method: string
            method of averaging the noise.
            Choices:
             'mean': Mean
             'median': Median
             'logmexp': Exponential of the mean of the logarithm of PSD (default)

    Returns:
    -------
        mp: array
            mean psd
    """

    if method == 'mean':
        mp = np.sqrt(np.mean(np.divide(y, 2), axis=-1))
    elif method == 'median':
        mp = np.sqrt(np.median(np.divide(y, 2), axis=-1))
    else:
        mp = np.log(np.divide((y + 1e-10), 2))
        mp = np.mean(mp, axis=-1)
        mp = np.exp(mp)
        mp = np.sqrt(mp)

    return mp




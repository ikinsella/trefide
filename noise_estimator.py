import numpy as np
import scipy as sp

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



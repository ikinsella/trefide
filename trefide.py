import cvxpy as cp
import numpy as np
import scipy as sp
from utils import butter_highpass

import matplotlib.pyplot as plt
"""Noise Estimation"""


def difference_operator(len_signal):
    # Gen Diff matrix
    diff_mat = (np.diag(2 * np.ones(len_signal), 0) +
                np.diag(-1 * np.ones(len_signal - 1), 1) +
                np.diag(-1 * np.ones(len_signal - 1), -1)
                )[1:len_signal - 1]
    return diff_mat


def _fft_estimator(signal, freq_range=[0.25, 0.5], max_samples=3072):
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


def _pwelch_estimator(signal, freq_range=[0.25, 0.5]):
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


def _boot_estimator(signal, num_samples=1000, len_samples=25):
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
        'fft': lambda x: _fft_estimator(x,
                                        freq_range=freq_range,
                                        max_samples=max_samples_fft),
        'pwelch': lambda x: _pwelch_estimator(x,
                                              freq_range=freq_range),
        'boot': lambda x: _boot_estimator(x,
                                          num_samples=num_samples_boot,
                                          len_samples=len_samples_boot)
    }[estimator]

    # Compute & return estimate of standard deviations for each signal
    return np.sqrt([summarizer(estimator(signal)) for signal in signals])


"""Activity detection/Region Partitioning"""


def _keep_min_consec(seq, min_consec=2):
    """Only keep positive elements which occur in sequences of at least min_sample
    consecutive positive elements."""
    idx = np.argwhere(seq > 0)
    keep_idx = np.zeros(len(idx), dtype=bool)
    for reach in np.arange(min_consec):
        keep_idx_tmp = np.ones(len(idx), dtype=bool)
        for inc in np.arange(min_consec):
            keep_idx_tmp = np.logical_and(keep_idx_tmp,
                                          np.in1d(idx - reach + inc, idx))
        keep_idx = np.logical_or(keep_idx,
                                 keep_idx_tmp)
    seq[idx[~keep_idx]] = 0
    return seq


def _connect_regions(spike_loc, active_min_gap, active_buffer):
    """ Connect breaches which occur within active_min_gap of one another into 
    single contiguous regions and add a buffer to pad both sides of active regions"""

    # Locate all breaches
    spike_loc_idx = np.argwhere(spike_loc > 0)

    # Avoid Case With No Spikes Detected
    if np.sum(spike_loc_idx) == 0:
        return [np.arange(len(spike_loc))], np.array([0.0])

    # Connect spike_loces in close proximity to one another
    # if spike_loc_idx[0] < active_min_gap:
    #    spike_loc[0:spike_loc_idx[0] + active_buffer] = 1
    for idx in np.arange(len(spike_loc_idx) - 1):
        if spike_loc_idx[idx + 1] - spike_loc_idx[idx] < active_min_gap:  # no transition
            spike_loc[np.arange(max(0, spike_loc_idx[idx] - active_buffer),
                                min(len(spike_loc), spike_loc_idx[idx + 1] + active_buffer))] = 1

    # Parse Regions into indexing arrays
    is_active = []
    rdx = 0
    regions = []
    rdx_start = 0
    while rdx_start < len(spike_loc):
        is_active.append(float(spike_loc[rdx_start]))
        try:
            rdx_end = rdx_start + \
                np.argwhere(~(spike_loc[rdx_start:] == is_active[rdx]))[0][0]
        except IndexError:
            rdx_end = len(spike_loc)
        regions.append(np.arange(rdx_start, rdx_end))
        rdx += 1
        rdx_start = rdx_end
    return regions, np.array(is_active)


def detect_regions(signal,
                   stdv=None,
                   noise_estimator='pwelch',
                   noise_summarizer='logmexp',
                   noise_max_samples_fft=3072,
                   noise_freq_range=[0.25, 0.5],
                   noise_num_samples_boot=1000,
                   noise_len_samples_boot=25,
                   filter_cutoff=100,
                   filter_fs=10000.0,
                   filter_order=6,
                   thresh_min_consec=2,
                   thresh_min_pnr=5,
                   active_min_gap=400,
                   active_buffer=50,
                   active_discount=0,
                   plot_en=False):
    """
    Partition input signal into regions dominated by spiking activity and 
    subthreshold activity. Active regions are detected by thresholding the 
    highpass activity (extracted with butterworth filter) of the input signal.
    ________
    Input:
        signal: (len_signal,) np.ndarray
            Collection of (gaussian) noise contaminated temporal signals 
            (required)
        stdv: positive integer
            Standard deviation of (gaussian noise) contaminating input signal 
            (default: estimate noise level with specified params)
        noise_estimator: string
            Method of estimating the noise level (when stdv is unspecified)
            Choices:
                'pwelch': average over high frequency components of Welch's
                          PSD estimate
                'fft': average over high frequency components of the FFT
                'boot': bootstrap estimates of the mse of linear fits to small
                        subsamples of the signal (only appropriate when signal
                        is approximately piecewise linear)
            (default: 'pwelch')
        noise_summarizer: string
            Method of averaging the power spectrum/bootstrap samples.
            Choices:
                'mean': Mean
                'median': Median
                'logmexp': Exponential of the mean of the logs 
            (default: 'logmexp')
        noise_freq_range: (2,) np.ndarray or len 2 list of increasing elements
                    between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power
            spectrum is averaged in the 'pwelch' and 'fft' noise estimators
            (default: [0.25,0.5])
        noise_max_samples_fft: positive integer
            Maximum number of samples which will be used in computing the
            power spectrum in the 'fft' noise estimator
            (default: 3072)
        noise_num_samples_boot: positive integer
            Number of bootstrapped estimates of MSE to average over in the
            'boot' estimator 
            (default: 1000)
        noise_len_samples_boot: positive integer < len_signal
            Length of subsampled signals from which MSE estimated are
            generated in the 'boot' estimator
            (default: 25)
        filter_cutoff: positive integer
            cutoff parameter in butterworth filter used to separate highpass
            signal content for thresholding
            (default: 100)
        filter_fs: positive integer
            Sampling frequency (Hz) parameter in butterworth filter used to
            separate highpass signal content for thresholding
            (default: 10000.0)
        filter_order: positive integer
            Order of butterworth filter used to separate highpass signal content
            for thresholding
            (default: 6)
        thresh_min_consec: positive integer
            While thresholding to locate spiking activity, only count threshold
            crossings which consist of thresh_'min_consec samples' consecutive
            samples
            (default: 2)
        thresh_min_pnr: positive integer
            The peak-to-noise ratio used to define the threshold for detecting
            potential spiking activity
            (default: 5),
        active_min_gap: positive integer
            Regions of spiking activity separated by less than 'active_min_gap'
            samples of inactivity are combined into a single continguous active
            region
            (default: 400)
        active_buffer: positive integer
            Append 'active_buffer' samples to each end of detected threshold 
            crossings while defining regions of spiking activity 
            (default: 50)
        active_discount: float in the interval [0,1)
            While enforcing noise constrain, shrink the noise constraint by 
            'active_discount' over regions where spiking activity was detected
            (e.g. 0 is no shrinkage and .05 shrinks stdv by 5%)
            (default: 0)
    ________
    Output:
        region_indices: len num_regions list of np.ndarrays
            Indices corresponding to each region
        fudge_factors: (num_regions,) np.ndarray
            Fudge factors applied to each region when enforcing the noise
            constraint ('1' for inactive regions, '1 - active_discount' for
            active regions)
    """

    # Auto detect noise level
    if stdv is None:
        stdv = estimate_noise([signal],
                              estimator=noise_estimator,
                              summarize=noise_summarizer,
                              freq_range=noise_freq_range,
                              max_samples_fft=noise_max_samples_fft,
                              num_samples_boot=noise_num_samples_boot,
                              len_samples_boot=noise_len_samples_boot)[0]

    # High pass filter to upweight spikes / remove slow fluctuations
    highpass = butter_highpass(signal,
                               cutoff=filter_cutoff,
                               fs=filter_fs,
                               order=filter_order)

      # Detect spike_loces of threshold with at least min_sample consec elements
    spike_loc = _keep_min_consec(np.abs(highpass / stdv) > thresh_min_pnr,
                                 min_consec=thresh_min_consec)

    if plot_en:
        plt.plot(spike_loc,'--r')
        plt.plot(signal/signal.max())
        plt.show()

    # todo: If there are too many spike_loces (min_pnr set too low),
    # fallback to reasonable default (single region?)

    # Group and buffer detected spike_loces into regions
    region_indices, is_active = _connect_regions(spike_loc,
                                                 active_min_gap=active_min_gap,
                                                 active_buffer=active_buffer)
    if plot_en:
        for reg_ in region_indices:
            plt.plot(highpass)
            plt.plot(reg_.flatten(),highpass[reg_.flatten(),],'--r')
            plt.show()

    # Compute fudge factor for each region
    fudge_factors = np.ones(len(is_active)) - active_discount * is_active

    return region_indices, fudge_factors


"""Trend Filtering"""


def constrained_l1tf(signal,
                     diff_mat=None,
                     stdv=None,
                     noise_estimator='pwelch',
                     noise_summarizer='logmexp',
                     noise_max_samples_fft=3072,
                     noise_freq_range=[0.25, 0.5],
                     noise_num_samples_boot=1000,
                     noise_len_samples_boot=25,
                     region_indices=None,
                     region_fudge_factors=None,
                     region_filter_cutoff=100,
                     region_filter_fs=10000.0,
                     region_filter_order=6,
                     region_thresh_min_consec=2,
                     region_thresh_min_pnr=5,
                     region_active_discount=0,
                     region_active_buffer=50,
                     region_active_min_gap=400,
                     solver='SCS',
                     lagrange_scaled =False,
                     verbose=False,
                     plot_en=False):
    """
    Denoise a single input signal y by applying the L1-Trend Filtering objective

       y_hat = argmin_{x} ||Dx||_1 
               s.t. ||y[idx] - x[idx]||_2 <= stdv * sqrt(len(idx)) * fudge_factor[idx]
               for idx in region_indices

    where D denotes a discrete difference operator, stdv the standard 
    deviation of the input signal noise, and region_indices a partition of the 
    input signal into disjoint segments over each of which a different noise 
    constraint is enforced. 
    ________
    Input:
        signal: (len_signal,) np.ndarray
            Collection of (gaussian) noise contaminated temporal signals 
            (required)
        diff_mat: (_, len_signal) np.ndarray
            Difference operator used in L1 trend filtering objective function
            (default: discrete second order difference operator)
        stdv: positive integer
            Standard deviation of (gaussian noise) contaminating input signal 
            (default: estimate noise level with specified params)
        noise_estimator: string
            Method of estimating the noise level (when stdv is unspecified)
            Choices:
                'pwelch': average over high frequency components of Welch's
                          PSD estimate
                'fft': average over high frequency components of the FFT
                'boot': bootstrap estimates of the mse of linear fits to small
                        subsamples of the signal (only appropriate when signal
                        is approximately piecewise linear)
            (default: 'pwelch')
        noise_summarizer: string
            Method of averaging the power spectrum/bootstrap samples.
            Choices:
                'mean': Mean
                'median': Median
                'logmexp': Exponential of the mean of the logs 
            (default: 'logmexp')
        noise_freq_range: (2,) np.ndarray or len 2 list of increasing elements
                    between 0 and 0.5
            Range of frequencies compared to Nyquist rate over which the power
            spectrum is averaged in the 'pwelch' and 'fft' noise estimators
            (default: [0.25,0.5])
        noise_max_samples_fft: positive integer
            Maximum number of samples which will be used in computing the
            power spectrum in the 'fft' noise estimator
            (default: 3072)
        noise_num_samples_boot: positive integer
            Number of bootstrapped estimates of MSE to average over in the
            'boot' estimator 
            (default: 1000)
        noise_len_samples_boot: positive integer < len_signal
            Length of subsampled signals from which MSE estimated are
            generated in the 'boot' estimator
            (default: 25)
        region_indices: len num_regions list of np.ndarrays
            Indices to define regions over which the noise constraint is 
            enforced (if used must also supply 'region_fudge_factors' 
            (default: auto-define region_indices and region_fudge_factors with 
             'detect_regions' function)
        region_fudge_factors: (num_regions,) np.ndarray
            Fudge factors applied to stdv while enfocing noise constrain over 
            each region supplied by the 'region_indices' 
            (default: np.ones(num_regions) i.e. enforce full noise constraint)
        region_filter_cutoff: positive integer
            cutoff parameter in butterworth filter used to separate highpass
            signal content for thresholding
            (default: 100)
        region_filter_fs: positive integer
            Sampling frequency (Hz) parameter in butterworth filter used to
            separate highpass signal content for thresholding
            (default: 10000.0)
        region_filter_order: positive integer
            Order of butterworth filter used to separate highpass signal content
            for thresholding
            (default: 6)
        region_thresh_min_consec: positive integer
            While thresholding to locate spiking activity, only count threshold
            crossings which consist of thresh_'min_consec samples' consecutive
            samples
            (default: 2)
        region_thresh_min_pnr: positive integer
            The peak-to-noise ratio used to define the threshold for detecting
            potential spiking activity
            (default: 5),
        region_active_min_gap: positive integer
            Regions of spiking activity separated by less than 
            'region_active_min_gap' samples of inactivity are combined into a 
            single continguous active region
            (default: 400)
        region_active_buffer: positive integer
            Append 'active_buffer' samples to each end of detected threshold 
            crossings while defining regions of spiking activity 
            (default: 50)
        region_active_discount: float in the interval [0,1)
            While enforcing noise constrain, shrink the noise constraint by 
            'active_discount' over regions where spiking activity was detected
            (e.g. 0 is no shrinkage and .05 shrinks stdv by 5%)
            (default: 0)
        solver: CVX optimizer used to solve trend filtering optimization
            Choices: 'SCS, 'ECOS', ...
            (default: 'SCS')
    ________
    Output:
        filtered_signal: (len_signal,) np.ndarray
            Indices corresponding to each region
        region_indices: len num_regions list of np.ndarrays
            Indices corresponding to each region
        lambdas: (num_regions,) np.ndarray
            Optimal Lagrange multipliers used to link constrained and lagrangian
            formulations of the trend filtering optimization
    """

    print('Auto detect noise level') if verbose else 0
    if stdv is None:
        stdv = estimate_noise([signal],
                              estimator=noise_estimator,
                              summarize=noise_summarizer,
                              freq_range=noise_freq_range,
                              max_samples_fft=noise_max_samples_fft,
                              num_samples_boot=noise_num_samples_boot,
                              len_samples_boot=noise_len_samples_boot)[0]

    print('Noise range [%d,%d]'%(stdv.max(),stdv.min())) if verbose else 0

    # Gen Diff matrix
    len_signal = len(signal)
    if diff_mat is None:
        diff_mat = (np.diag(2 * np.ones(len_signal), 0) +
                    np.diag(-1 * np.ones(len_signal - 1), 1) +
                    np.diag(-1 * np.ones(len_signal - 1), -1)
                    )[1:len_signal - 1]

    print('Auto-detect relevant regions') if verbose else 0
    if region_indices is None:
        region_indices, fudge_factors = detect_regions(signal,
                                                       stdv=stdv,
                                                       filter_cutoff=region_filter_cutoff,
                                                       filter_fs=region_filter_fs,
                                                       filter_order=region_filter_order,
                                                       thresh_min_consec=region_thresh_min_consec,
                                                       thresh_min_pnr=region_thresh_min_pnr,
                                                       active_min_gap=region_active_min_gap,
                                                       active_buffer=region_active_buffer,
                                                       active_discount=region_active_discount)
    elif fudge_factors is None:
        fudge_factors = np.ones(len(region_indices))

    print (fudge_factors) if verbose else 0
    print('Translate trend filtering optimization to CVX') if verbose else 0
    filtered_signal = cp.Variable(len_signal)
    objective = cp.Minimize(cp.norm(diff_mat*filtered_signal, 1))
    constraints = [cp.norm(signal[idx] - filtered_signal[idx], 2)
                   <= fudge * stdv * np.sqrt(len(idx))
                   for idx, fudge in zip(region_indices, fudge_factors)]
    # Solve CVX
    cp.Problem(objective, constraints).solve(solver=solver,
                    max_iters=1000,verbose=False)  # 'ECOS' or 'SCS'
    
    lambdas = [constraint.dual_value for constraint in constraints]
    if lagrange_scaled: # scale by fudge factor for closed iterations:
        lambdas = [fudge_factors[ii]/lambda_ if lambda_ !=0 else lambda_ for ii, 
                   lambda_ in enumerate(lambdas)]
    return np.asarray(filtered_signal.value).flatten(), region_indices, lambdas


def denoise(signals,
            stdvs=None,
            diff_mat=None,
            noise_estimator='pwelch',
            noise_summarizer='logmexp',
            noise_max_samples_fft=3072,
            noise_freq_range=[0.25, 0.5],
            noise_num_samples_boot=1000,
            noise_len_samples_boot=25,
            region_indices=None,
            region_fudge_factors=None,
            region_filter_cutoff=100,
            region_filter_fs=10000.0,
            region_filter_order=6,
            region_thresh_min_consec=2,
            region_thresh_min_pnr=5,
            region_active_discount=0,
            region_active_buffer=50,
            region_active_min_gap=400,
            solver='SCS'):
    """
    Apply trend filtering denoiser to collection of signals
    """

    # Default to discrete second order difference operator
    if diff_mat is None:
        len_signals = signals.shape[1]
        diff_mat = (np.diag(2 * np.ones(len_signals), 0) +
                    np.diag(-1 * np.ones(len_signals - 1), 1) +
                    np.diag(-1 * np.ones(len_signals - 1), -1)
                    )[1:len_signals - 1]

    # Compute noise standard deviation estimates for each signal
    if stdvs is None:
        stdvs = estimate_noise(signals,
                               estimator=noise_estimator,
                               summarize=noise_summarizer,
                               freq_range=noise_freq_range,
                               max_samples_fft=noise_max_samples_fft,
                               num_samples_boot=noise_num_samples_boot,
                               len_samples_boot=noise_len_samples_boot)

    # Trend filter each signal
    return np.array([constrained_l1tf(signal,
                                      diff_mat=diff_mat,
                                      stdv=stdv,
                                      region_indices=region_indices,
                                      region_fudge_factors=region_fudge_factors,
                                      region_filter_cutoff=region_filter_cutoff,
                                      region_filter_fs=region_filter_fs,
                                      region_filter_order=region_filter_order,
                                      region_thresh_min_pnr=region_thresh_min_pnr,
                                      region_thresh_min_consec=region_thresh_min_consec,
                                      region_active_discount=region_active_discount,
                                      region_active_buffer=region_active_buffer,
                                      region_active_min_gap=region_active_min_gap,
                                      solver=solver)[0]
                     for signal, stdv in zip(signals, stdvs)])

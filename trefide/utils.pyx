# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np

# --------------------------------------------------------------------------- #
# -------------------- Temporal Signal Noise Estimation --------------------- #
# --------------------------------------------------------------------------- #

cdef extern from "math.h":
    double floor(double) nogil

cdef extern from "trefide.h":
    double _psd_noise_estimate "psd_noise_estimate" (const size_t N, 
                                                     const double *x, 
                                                     void* FFT) nogil

    void welch(const size_t N, 
               const int L, 
               const int R, 
               const double fs, 
               const double* x, 
               double* psd,
               void* FFT) nogil


cpdef double[:, ::1] welch_psd_estimate(double[:, ::1] signal, 
                                        int nsamp_seg=256, 
                                        int nsamp_overlap=128,
                                        double fs=1):
    """
    Estimates the Power Spectral Density (PSD) using Welch's method of
    controlling variance by averaging over lower resolution PSD estimates 
    from smaller, overlapping signal segments.
    ________
    Input:
        signal: (nchan, nsamp) np.ndarray  (required)
            Noise contaminated temporal signal        
        nsamp_seg: int (optional) 
            Number of samples in each segment used for individual PSD estimates
        nsamp_overlap: int (optional)
            Number of samples of overlap between consecutive segments
        fs: double (optional)
            Sampling frequency of input signal
        (TODO: modify cpp implementation for more windowing options (currently Hann)
    ________
    Output:
        Pxx: (nchan_signal, floor(nsamp_seg / 2) + 1) np.ndarray
            Estimates of PSD coefficients for each input channel
    """

    # Declare & Initialize Local Variables
    cdef size_t c
    cdef size_t nchan = signal.shape[0]
    cdef size_t nsamp = signal.shape[1] 
    cdef size_t ncoef = <size_t> floor(nsamp / 2) + 1

    # Allocate & Init PSD Coefs (IMPORTANT: Pxx must be init'd to 0)
    cdef double[:, ::1] pxx = np.zeros((nchan, ncoef), dtype=np.float64)

    # Compute & Return Welch's PSD Estimate (Pxx modified inplace)
    # TODO: modify cpp implementation for more efficient multichannel estimation
    with nogil:
        for c in range(nchan):
            welch(nsamp, nsamp, nsamp, fs, &signal[c,0], &pxx[c,0], NULL) 
    return pxx


cpdef double[::1] psd_noise_estimate(double [:,::1] signal):
    """
    Estimates the variance of the (assumed to be gaussian) noise
    contaminating an input signal by averaging over the high frequency 
    components of Welch's PSD Estimate 
    ________
    Input:
        signal: (nchan, nsamp) np.ndarray 
            Noise contaminated temporal signal (required)
        (TODO: modify cpp implementation allowing different PSD params)
        (TODO: modify cpp implementation to allow different types of averaging) 
        (TODO: modify cpp implementation to allow input of different freq 
               ranges for averaging ... currently [.25,.5]) 
    ________
    Output:
        vars: (nchan_signal,) np.ndarray
            Estimates of the noise variance contaminating each input channel
    """

    # Declare & Initialize Local Variables
    cdef size_t c
    cdef size_t nchan = signal.shape[0]
    cdef size_t nsamp = signal.shape[1]

    # Allocate Space For Output Variances
    cdef double[::1] var_hat = np.empty(nchan, dtype=np.float64)

    # Compute & Return Estimates 
    # TODO: modify cpp implementation for more efficient multichannel estimation
    with nogil:
        for c in range(nchan):
            var_hat[c] = _psd_noise_estimate(nsamp, &signal[c,0], NULL)
    return var_hat 

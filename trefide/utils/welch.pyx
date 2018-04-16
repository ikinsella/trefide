# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False


cdef extern from "src/welch.c":
    double _psd_noise_estimate "psd_noise_estimate" (const size_t n,
                                                     const double *x) nogil



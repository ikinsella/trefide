cimport numpy as np
import numpy as np


cdef class TrendFilter:
    """ """
    cdef int notfit, verbose, maxiter
    cdef public np.intp_t T
    cdef public np.double_t[::1] warm_start, weights
    cdef public double lambda_, delta
    cpdef double[::1] _fit(self, double[::1] y)
    cpdef double[::1] denoise(self, double[::1] y, int refit=*)

cimport numpy as np
import numpy as np


cdef class TrendFilter:
    """ """
    cdef int notfit, verbose
    cdef np.intp_t maxiter
    cdef public np.intp_t T
    cdef public np.double_t[::1] warm_start, weights
    cdef public double lambda_, delta
    cpdef np.double_t[::1] _fit(self, np.double_t[::1] y)
    cpdef np.double_t[::1] denoise(self, np.double_t[::1] y, int refit=*)

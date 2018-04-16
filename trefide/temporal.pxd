cimport numpy as np
import numpy as np


cdef class TrendFilter:
    """ """
    cdef int notfit, verbose
    cdef size_t maxiter
    cdef public size_t T
    cdef public double[::1] warm_start, weights
    cdef public double lambda_, delta
    cpdef double[::1] _fit(self, double[::1] y, double delta=?)
    cpdef double[::1] denoise(self, double[::1] y, int refit=?, double delta=?)

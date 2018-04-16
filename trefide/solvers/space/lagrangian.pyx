# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np
cimport cython

from cpython cimport bool

np.import_array()


cdef extern from "proxtv.h":
    int DR2_TV(size_t M, 
               size_t N, 
               double*unary, 
               double W1, 
               double W2, 
               double norm1, 
               double norm2, 
               double*s, 
               int nThreads, 
               int maxit, 
               double* info) nogil


cpdef double[::1] ldr2_tv(const size_t d1,
                          const size_t d2,
                          const double[::1] y,
                          const double lambda_tv):
    """ Handle to weighted pdas solver allowing warm start intialization"""
   
    # Initialize variables
    cdef int status
    cdef double[::1] y_hat = np.empty(d1*d2, dtype=np.float64)
    cdef double[::1] info = np.empty(3, dtype=np.float64)

    # Call weighted pdas C routine 
    with nogil:
        status =DR2_TV(d1, 
                       d2, 
                       &y[0], 
                       lambda_tv, 
                       lambda_tv, 
                       1, 
                       1, 
                       &y_hat[0], 
                       1, 
                       1, 
                       &info[0])     

    # Check Convergence Status 
    if status < 0:
        raise RuntimeError("Total Variation Solver Failed.")

    return y_hat

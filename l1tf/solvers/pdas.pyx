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

cdef extern from "C_pdas/pdas_sg2.c":
    int call_active_set "active_set" (const int n,
				      const double *y,
				      const double lambda_,
				      double *x,
				      double *z,
				      double p,
				      const int m,
				      const double delta_s,
				      const double delta_e,
				      const int maxiter,
				      const int verbose) nogil

def solve(double[::1] y,
          double lambda_,
	  const int maxiter,
	  const int verbose):
    """
    Solve L1 trend filter via primal-dual interior point method

    minimize rescale*||data_-x||_2 ^2 + lambda ||z||_1
    subject to  z=Dx

    Parameters:
    ----------
    data_:      np.array (L,)
                data sequence 
    lambda_:    float
                regularization parameter
    max_lambda: boolean
                find the upper bound for regularization parameter
                scale this value by given lambda_
    rescale:    float
                scaling parameter for L2 norm
    verbose:    int {0,1}
                flag
    Output:
    -------
    data_hat :  np.array (L,)
                data sequence
    """
    
    cdef np.intp_t n = y.shape[0]
    cdef np.double_t[::1] x_hat = np.empty(n, dtype=np.double)
    cdef np.double_t[::1] z_hat = np.zeros(n - 2, dtype=np.double)
    cdef double p = 1    
    cdef int m = 5
    cdef double delta_s = .9
    cdef double delta_e = 1.1
    cdef int iter_status
    
    with nogil:
        iter_status = call_active_set(n,
                                      &y[0],
                                      lambda_,
                                      &x_hat[0],
				      &z_hat[0],
				      p,
				      m,
				      delta_s,
				      delta_e,
                                      maxiter,
                                      verbose)

    if iter_status < 0:
        raise RuntimeError("Active Set Failed To Converge")

    return x_hat, z_hat


def warm_start(double[::1] y,
	       double lambda_,
	       double[::1] z_hat,
	       const int maxiter,
	       const int verbose):
    """
    Solve L1 trend filter via primal-dual interior point method

    minimize rescale*||data_-x||_2 ^2 + lambda ||z||_1
    subject to  z=Dx

    Parameters:
    ----------
    data_:      np.array (L,)
                data sequence 
    lambda_:    float
                regularization parameter
    max_lambda: boolean
                find the upper bound for regularization parameter
                scale this value by given lambda_
    rescale:    float
                scaling parameter for L2 norm
    verbose:    int {0,1}
                flag
    Output:
    -------
    data_hat :  np.array (L,)
                data sequence
    """
    
    cdef np.intp_t n = y.shape[0]
    cdef np.double_t[::1] x_hat = np.empty(n, dtype=np.double)
    cdef double p = 1    
    cdef int m = 5
    cdef double delta_s = .9
    cdef double delta_e = 1.1
    cdef int iter_status
    
    with nogil:
        iter_status = call_active_set(n,
                                      &y[0],
                                      lambda_,
                                      &x_hat[0],
				      &z_hat[0],
				      p,
				      m,
				      delta_s,
				      delta_e,
                                      maxiter,
                                      verbose)

    if iter_status < 0:
        raise RuntimeError("Active Set Failed To Converge")

    return x_hat, z_hat

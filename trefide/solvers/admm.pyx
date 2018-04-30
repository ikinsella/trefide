# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import warnings
cimport cython

import numpy as np
cimport numpy as np

from libcpp cimport bool

np.import_array()


# ---------------------------------------------------------------------------- #
# ------------------------------ ADMM Wrappers ------------------------------- #
# ---------------------------------------------------------------------------- #

cdef extern from "trefide.h":
    
    short constrained_tf_admm(const int n,           # data length
                              double* x,             # data locations
                              double *y,             # data observations
                              double *w,             # data observation weights
                              const double delta,    # MSE constraint (noise var estimate)	
                              double *beta,          # primal variable
                              double *alpha,
                              double *u,
                              double *lambda_,       # initial regularization parameter
                              int *iters,            # pointer to iter # (so we can return it)
                              const double tol,      # relative difference from target MSE
                              const int verbose) nogil

    short langrangian_tf_admm(const int n,           # data length
                              double* x,             # data locations
                              double *y,             # data observations
                              double *w,             # data observation weights
                              double lambda_,        # regularization parameter
                              double *beta,          # primal variable
                              double *alpha,
                              double *u,
                              int *iter_,            # pointer to iter # (so we can return it)
                              const int verbose) nogil


# ---------------------------------------------------------------------------- #
# --------------------------- ADMM Solver Wrappers --------------------------- #
# ---------------------------------------------------------------------------- #


cpdef cps_admm(const double[::1] y,        # Observations
               const double delta,         # MSE constraint
               double[::1] w=None,        # Observation weights
               double[::1] alpha=None,     # Dual variable warm start
               double[::1] u=None,     # Dual variable warm start
               double lambda_=-1,          # Lagrange multiplier warm start
               double tol=5e-2,            # Constraint tolerance
               int verbose=0):
    """ 
    Shallow wrapper to call libtrefide solver 
    """
    
    # Declare & Intialize Local Variables
    cdef int iters = 0;
    cdef short status;
    cdef size_t t = y.shape[0]

    # Allocate Space For Output
    cdef double[::1] x = np.arange(t, dtype=np.float64)
    cdef double[::1] beta = y.copy()

    # Default to unweighted l1tf
    if w is None:
        w = np.ones(t, dtype=np.float64)

    # Default to initializing dual vars at 0
    if alpha is None:
        alpha = np.zeros(t , dtype=np.float64)  # Allocate extra buffer
    if u is None:
        u = np.zeros(t , dtype=np.float64)  # Allocate extra buffer
 
    # Call Solver
    with nogil:
        status  = constrained_tf_admm(t, &x[0], &y[0], &w[0], delta, &beta[0],
                                      &alpha[0], &u[0], &lambda_, &iters,
                                      tol, verbose)

    # Check For Failures 
    if status < 0:
        raise RuntimeError("ADMM Failed Within CPS Line Search.")
    elif status == 0:
        warnings.warn("CPS ADMM line search stalled or exceeded maxiter.")
 
    return np.asarray(beta), np.asarray(alpha), np.asarray(u), lambda_
 

cpdef ladmm(const double[::1] y,        # Observations
            const double lambda_,         # MSE constraint
            double[::1] w=None,        # Observation weights
            double[::1] alpha=None,     # Dual variable warm start
            double[::1] u=None,     # Dual variable warm start
            int verbose=0):
    """ Handle to weighted pdas solver allowing warm start intialization"""
   
    # Declare & Intialize Local Variables
    cdef int iter_ = 0;
    cdef short status;
    cdef size_t t = y.shape[0]

    # Allocate Space For Output
    cdef double[::1] x = np.arange(t, dtype=np.float64)
    cdef double[::1] beta = y.copy()

    # Default to unweighted l1tf
    if w is None:
        w = np.ones(t, dtype=np.float64)

    # Default to initializing dual vars at 0
    if alpha is None:
        alpha = np.zeros(t , dtype=np.float64)  # Allocate extra buffer
    if u is None:
        u = np.zeros(t , dtype=np.float64)  # Allocate extra buffer
 
    # Call Solver
    with nogil:
        status = langrangian_tf_admm(t, &x[0], &y[0], &w[0], lambda_, &beta[0],
                                     &alpha[0], &u[0], &iter_, verbose)

    # Check For Failures 
    if status < 0:
        raise RuntimeError("ADMM Failed Within CPS Line Search.")
 
    return np.asarray(beta), np.asarray(alpha), np.asarray(u)

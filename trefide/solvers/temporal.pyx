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
# ----------------------------- c++ lib imports ------------------------------ #
# ---------------------------------------------------------------------------- #


cdef extern from "math.h":
    double sqrt(double m) nogil
    double log(double m) nogil
    double exp(double m) nogil
    double pow(double m, int n) nogil


cdef extern from "trefide.h":

    short constrained_wpdas(const int n,
                            const double *y,
                            const double *wi,
                            const double delta,
                            double *x,
                            double *z,
                            double *lambda_,
                            int *iters,
                            const int max_interp,
                            const double tol,
                            const int verbose) nogil

    short weighted_pdas (const int n,
                         const double *y,
                         const double *wi,
                         const double lambda_,
                         double *x,
                         double *z,
                         int *iters,
                         double p,
                         const int m,
                         const double delta_s,
                         const double delta_e,
                         const int maxiter,
                         const int verbose) nogil

    int l1tf (const int n,
              const double *y,
              const double lambda_,
              double *x,
              double *z,
              int *iter_,
              const double tol,
              const int maxiter,
              const int verbose) nogil

    double l1tf_lambdamax (const int n,
			   double *y,
			   const int verbose) nogil
 

# ---------------------------------------------------------------------------- #
# -------------------- Primal Dual Active Set Wrappers ----------------------- #
# ---------------------------------------------------------------------------- #


cpdef cpdas(const double[::1] y,        # Observations
            const double delta,         # MSE constraint
            double[::1] wi=None,        # Observation weights
            double[::1] z_hat=None,     # Dual variable warm start
            double lambda_=-1,          # Lagrange multiplier warm start
            int max_interp=1,           # Number of interps before stepping
            double tol=1e-3,            # Constraint tolerance
            int verbose=0):
    """ 
    Shallow wrapper to call libtrefide solver 
    """
    
    # Declare & Intialize Local Variables
    cdef int iters;
    cdef short status;
    cdef size_t t = y.shape[0]

    # Allocate Space For Output
    cdef double[::1] x_hat = np.empty(t, dtype=np.float64)

    # Default to unweighted l1tf
    if wi is None:
        wi = np.ones(t, dtype=np.float64)

    # Default to initializing dual var at 0
    if z_hat is None:
        z_hat = np.zeros(t - 2, dtype=np.float64)
 
    # Call Solver
    with nogil:
        iters = constrained_wpdas(t, &y[0], &wi[0], delta, &x_hat[0], &z_hat[0],
                                  &lambda_, &iters, max_interp, tol, verbose)

    # Check For Failures 
    if iters < 0:
        raise RuntimeError("CPDAS Solver Failed.")
    elif iters == 0:
        warnings.warn("CPDAS line search stalled. Returning best solution found.")
 
    return x_hat, z_hat, lambda_, iters

 
cpdef lpdas(double[::1] y,
            const double lambda_,
	    double[::1] wi=None,
	    double[::1] z_hat=None,
            double p=1,    
            int m=5,
            double delta_s=.9,
            double delta_e=1.1,
	    int maxiter=2000,
	    int verbose=0):
    """ Handle to weighted pdas solver allowing warm start intialization"""
   
    # Declare & Intialize Local Variables
    cdef int iters
    cdef short status
    cdef size_t n = y.shape[0]

    # Allocate Space For Output
    cdef double[::1] x_hat = np.empty(n, dtype=np.float64)

    # Default to unweighted loss
    if wi is None:
        wi = np.ones(n, dtype=np.float64)
    
    # Default warm start dual variable at 0
    if z_hat is None:
        z_hat = np.zeros(n - 2, dtype=np.float64)

    # Call weighted pdas C routine 
    with nogil:
        status = weighted_pdas(n, &y[0], &wi[0], lambda_,  &x_hat[0], &z_hat[0],
                               &iters, p, m, delta_s, delta_e, maxiter, verbose)
    
    # Check For Failures 
    if status < 0:
        raise RuntimeError("LPDAS Solver Failed.")
    elif status == 0:
        warnings.warn("LPDAS failed to converge in MAXITER iterations.\
                Returning solution for last partition evaluated.")

    return x_hat, z_hat, iters


# ---------------------------------------------------------------------------- #
# ------------------------------ ADMM Wrappers ------------------------------- #
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
# --------------------- Interion Point Method Wrappers ----------------------- #
# ---------------------------------------------------------------------------- #


def l1tf_lambda_max(double[::1] data_,
                    int verbose):
    """
    Upper bound for regularization parameter

    Parameters:
    ----------
    data_:      np.array (L,)
                data sequence 
    verbose:    int {0,1}
                flag

    Outputs:
    --------
    lambda_max: float
                lambda upper bound
    """

    cdef size_t data_length = data_.shape[0]
    cdef double lambda_max

    lambda_max = l1tf_lambdamax(data_length,
                                &data_[0],
                                verbose)

    if lambda_max < 0:
        raise RuntimeError("Lambda < 0")

    return lambda_max


cpdef ipm(double[::1] y,
          double lambda_,
          bool max_lambda=False,
          double tol=1e-4,
	  int maxiter=200,
          int verbose=0):
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
    
    cdef size_t n = y.shape[0]
    cdef double[::1] x_hat = np.empty(n, dtype=np.float64)
    cdef double[::1] z_hat = np.empty(n - 2, dtype=np.float64)    
    cdef double lambda_max
    cdef int iter_
    cdef int iter_status

    if max_lambda:
        lambda_max = l1tf_lambdamax(n,
                                    &y[0],
                                    verbose)
        if lambda_max < 0:
            raise RuntimeError("Lambda < 0")
        lambda_ *= lambda_max

    with nogil:
        iter_status = l1tf(n,
                           &y[0],
                           lambda_,
                           &x_hat[0],
                           &z_hat[0],
                           &iter_,
                           tol,
                           maxiter,
                           verbose)

    if iter_status < 0:
        raise RuntimeError("Interior Point Method Failed To Converge In MAXITER iterations.")
	
    return x_hat, z_hat, iter_

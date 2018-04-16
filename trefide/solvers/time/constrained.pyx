# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

np.import_array()


cdef extern from "math.h":
    double sqrt(double m) nogil
    double log(double m) nogil
    double exp(double m) nogil
    double pow(double m, int n) nogil


cdef extern from "trefide.h":
    int line_search (const int n,
                     const double *y,
                     const double *wi,
                     const double delta,
                     const double tau,
                     double *x,
                     double *z,
                     double *lambda_,
                     int *iters,
                     const int max_interp,
                     const double tol,
                     const int verbose) nogil


cpdef cpdas(const double[::1] y,        # Observations
            const double delta,         # MSE constraint
            double[::1] wi=None,        # Observation weights
            double[::1] z_hat=None,     # Dual variable warm start
            double lambda_=0,           # Lagrange multiplier warm start
            int step_part=60,           # Lagrange multiplier warm start
            int max_interp=1,           # Number of interps before stepping
            double tol=1e-3,            # Constraint tolerance
            int verbose=0):
    """ 
    Conduct line search in lambda space to solve constrianed problem 
    via lagrangian pdas solver
    """
    
    cdef size_t t = y.shape[0]
    cdef size_t i
    cdef double tau, delta_w, scale = 0
    cdef int iter_status, iters = 0
    cdef double[::1] x_hat = np.empty(t, dtype=np.float64)

    # Default to unweighted l1tf
    if wi is None:
        wi = np.ones(t, dtype=np.float64)
        delta_w = 1
    else: 
        delta_w = 0
        for i in range(t):
            delta_w += 1 / wi[i]
        delta_w /= t

    # Default to initializing dual var at 0
    if z_hat is None:
        z_hat = np.zeros(t - 2, dtype=np.float64)

    # Initialize step size as target interval width / step_partition
    with nogil:
        scale = compute_scale(t, &y[0], delta)
        tau = (log(20+(1/scale)) - log(3+(1/scale))) / 60
        # default to initializing lambda middle of target interval
        if lambda_ <= 0:
            lambda_ = exp((log(20+(1/scale)) - log(3+(1/scale))) / 2 + log(3*scale + 1)) - 1
    
    # Call Solver
    with nogil:
        iter_status = line_search(t,
                                  &y[0],
                                  &wi[0],
                                  delta*delta_w,
                                  tau,
                                  &x_hat[0],
                                  &z_hat[0],
                                  &lambda_,
                                  &iters,
                                  max_interp,
                                  tol,
                                  verbose)

    # Check Convergence 
    if iter_status < 0:
        raise RuntimeError("PDAS failed to converge in MAXITER iterations.")
 
    return x_hat, z_hat, lambda_, iters


cdef void cpdas_lite(const size_t t,
                     double delta,       # MSE constraint
                     const double *wi,               # Observation weights
                     double *x,                # Noisey Observations Denoise In Place
                     double *z,                # Dual variable warm start
                     double *lambda_,          # Lagrange multiplier warm start
                     int verbose) nogil:
    """ 
    Conduct line search in lambda space to solve constrianed problem 
    via lagrangian pdas solver
    """
    
    # Declare Local Variables
    cdef double *y = <double *> malloc(t * sizeof(double))
    cdef int iter_status, iters = 0
    cdef double scale, tau
    cdef size_t i

    # Copy x into y
    for i in range(t):
        y[i] = x[i]

    # Compute Ideal Step Size
    scale = compute_scale(t, y, delta)
    tau = (log(20 + (1 / scale)) - log(3 + (1 / scale))) / 60

    # Call Solver
    with nogil:
        iter_status = line_search(t,
                                  y,
                                  wi,
                                  delta,
                                  tau,
                                  x,
                                  z,
                                  lambda_,
                                  &iters,
                                  1,
                                  1e-3,
                                  verbose)
    # Release Allocated Memory
    free(y)


cdef double compute_scale(const size_t t, 
                          const double *y, 
                          const double delta) nogil:
    """ Compute Scaling Factor For 2nd Order TF"""

    # Declare Internal Variables
    cdef size_t i
    cdef double var_y = 0

    # Initialize step size as target interval width / step_partition
    for i in range(t):
        var_y += pow(y[i], 2)  # y assumed detrended and centered
    var_y /= t
    if var_y < delta:
        return sqrt(var_y) / sqrt(.1)
    else:
        return delta / sqrt(var_y - delta)

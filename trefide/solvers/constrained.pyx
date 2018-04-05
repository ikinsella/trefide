# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import math
import numpy as np
cimport numpy as np
cimport cython

from cpython cimport bool

np.import_array()


cdef extern from "src/line_search.c":
    int call_line_search "line_search" (const int n,
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

cpdef cpdas(const double[::1] y,    # Observations
        const double delta,         # MSE constraint
        double[::1] wi=None,        # Observation weights
        double[::1] z_hat=None,     # Dual variable warm start
        double lambda_=0,           # Lagrange multiplier warm start
        int step_part=60,     # Lagrange multiplier warm start
        int max_interp=1,     # Number of interps before stepping
        double tol=1e-3,      # Constraint tolerance
        int verbose=0):
    """ 
    Conduct line search in lambda space to solve constrianed problem 
    via lagrangian pdas solver
    """
    
    cdef np.intp_t n = y.shape[0]
    cdef double tau, int_width, scale = 0
    cdef int i, iter_status, iters = 0
    cdef np.double_t[::1] x_hat = np.empty(n, dtype=np.double)

    # Default to unweighted l1tf
    if wi is None:
        wi = np.ones(n, dtype=np.double)

    # Default to initializing dual var at 0
    if z_hat is None:
        z_hat = np.zeros(n - 2, dtype=np.double)

    # Initialize step size as target interval width / step_partition
    for i in range(n):
        scale += y[i] ** 2  # y assumed detrended or detrended
    scale /= n
    scale = delta / math.sqrt(scale - delta) 
    int_width = math.log(20+(1/scale)) - math.log(3+(1/scale))
    tau = int_width / 60

    # default to initializing lambda middle of target interval
    if lambda_ <= 0:
        lambda_ = math.exp(int_width / 2 + math.log(3*scale + 1)) - 1
    
    # Call Solver
    with nogil:
        iter_status = call_line_search(n,
				       &y[0],
				       &wi[0],
				       delta,
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

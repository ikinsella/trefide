# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

cimport numpy as np
import numpy as np
# cimport cython
# import cython
from trefide.solvers.lagrangian import lpdas
from trefide.solvers.constrained import cpdas 
from trefide.utils.noise import estimate_noise

np.import_array()

cdef class TrendFilter(object):
    """ """
    cdef int isfit, verbose, maxiter
    cdef public np.intp_t T
    cdef public np.double_t[::1] warm_start, weights
    cdef public double lambda_, delta

    def __init__(self, int T):
        self.isfit = 0
        self.T = T
        self.warm_start = np.zeros(self.T-2, dtype=np.double)
        self.weights = np.ones(self.T, dtype=np.double)
        self.maxiter=2000
        self.verbose=0

    cpdef double[::1] fit(self, np.ndarray y):  
        """ """
        # Estimate Noise
        self.delta = estimate_noise([y], summarize='mean')[0] ** 2
        # Call constrained solver
        x_hat, self.warm_start, self.lambda_, _ = cpdas(y, self.delta)
        self.isfit = 1
        return x_hat

    cpdef double[::1] predict(self, double[::1] y):
        """ """
        if self.isfit:
            x_hat, z_hat, _ = lpdas(y, 
                                    self.lambda_, 
                                    wi=self.weights, 
                                    z_hat=self.warm_start,  
                                    maxiter=self.maxiter, 
                                    verbose=self.verbose)
            self.warm_start = z_hat
            return x_hat
        else:
            return self.fit(y)
 

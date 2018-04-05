# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

cimport numpy as np
import numpy as np
from trefide.solvers.lagrangian import lpdas
from trefide.solvers.constrained import cpdas 
from trefide.utils.noise import estimate_noise

np.import_array()

cdef class TrendFilter(object):
    """ """
    cdef int notfit, verbose, maxiter
    cdef public np.intp_t T
    cdef public np.double_t[::1] warm_start, weights
    cdef public double lambda_, delta

    def __init__(self, int T, int maxiter=2000, int verbose=0):
        """ Initialze filter for signal of length T"""
        self.notfit = 1
        self.T = T
        self.warm_start = np.zeros(self.T-2, dtype=np.double)
        self.weights = np.ones(self.T, dtype=np.double)
        self.maxiter=maxiter
        self.verbose=verbose

    cpdef double[::1] _fit(self, double[::1] y):  
        """ Fit model parameters by solving constrained l1tf"""
        # Estimate Noise
        self.delta = estimate_noise([y], summarize='mean')[0] ** 2
        # Call constrained solver
        x_hat, self.warm_start, self.lambda_, _ = cpdas(y, 
                                                        self.delta,
                                                        wi=self.weights,
                                                        z_hat=self.warm_start, 
                                                        lambda_=self.lambda_,
                                                        verbose=self.verbose)
        self.notfit = 0
        return x_hat

    cpdef double[::1] denoise(self, double[::1] y, int refit=1):
        """ Denoise an input signal, default to constrained l1tf"""
        if refit or self.notfit:
            return self._fit(y)
        else:
            x_hat, z_hat, _ = lpdas(y, 
                                    self.lambda_, 
                                    wi=self.weights, 
                                    z_hat=self.warm_start,  
                                    maxiter=self.maxiter, 
                                    verbose=self.verbose)
            return x_hat
 

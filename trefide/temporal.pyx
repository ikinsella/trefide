# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

cimport numpy as np
import numpy as np
from trefide.solvers.temporal import cps_cpdas, lpdas
from trefide.utils import psd_noise_estimate


np.import_array()

cdef class TrendFilter:
    """ """
    
    
    def __cinit__(self, size_t T, size_t maxiter=2000, int verbose=0):
        """ Initialze filter for signal of length T"""
        self.notfit = 1
        self.T = T
        self.warm_start = np.zeros(self.T-2, dtype=np.float64)
        self.weights = np.ones(self.T, dtype=np.float64)
        self.maxiter=maxiter
        self.verbose=verbose


    cpdef double[::1] _fit(self, double[::1] y, double delta=0):  
        """ Fit model parameters by solving constrained l1tf"""
        # Estimate Noise
        if delta > 0:
            self.delta = delta
        else:
            self.delta = psd_noise_estimate(y[None, :])[0]

        # Call constrained solver
        x_hat, self.warm_start, self.lambda_, _ = cps_cpdas(y, 
                                                            self.delta,
                                                            wi=self.weights,
                                                            z_hat=self.warm_start, 
                                                            lambda_=self.lambda_,
                                                            verbose=self.verbose)
        self.notfit = 0
        return x_hat

    cpdef double[::1] denoise(self, double[::1] y, int refit=1, double delta=0):
        """ Denoise an input signal, default to constrained l1tf"""
        if refit or self.notfit:
            return self._fit(y, delta=delta)
        else:
            x_hat, z_hat, _ = lpdas(y, 
                                    self.lambda_, 
                                    wi=self.weights, 
                                    z_hat=self.warm_start,  
                                    maxiter=self.maxiter, 
                                    verbose=self.verbose)
            return x_hat 

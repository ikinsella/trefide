# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

cimport numpy as np
import numpy as np
# cimport cython
# import cython
import solvers.lagrange.pdas as pdas
import solvers.lagrange.wpdas as wpdas
import solvers.constrained as constrained

np.import_array()

cdef class TrendFilter:
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
        self.maxiter=1000
        self.verbose=1

    cpdef double[::1] fit(self, np.ndarray y):  
        """ """
        # Call constrained solver
        x_hat, z_hat, lams, _, _, _, delta = constrained.solve(y)
        # Set Params
        #self.warm_start = z_hat
        #self.lambda_ = lams[-1]
        #self.delta = delta
        #self.weights = np.ones(self.T)
        self.isfit = 1
        return x_hat

    cpdef predict(self, double[::1] y):
        """ """
        if self.isfit:
            x_hat, z_hat, _ = pdas.warm_start(y, 
                                              self.lambda_, 
                                              self.weights, 
                                              self.warm_start,  
                                              self.maxiter, 
                                              self.verbose)
            self.warm_start = z_hat
            return x_hat
        else:
            return self.fit(y)




        
        

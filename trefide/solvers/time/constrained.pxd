# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np


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
            double[::1] wi=?,        # Observation weights
            double[::1] z_hat=?,     # Dual variable warm start
            double lambda_=?,           # Lagrange multiplier warm start
            int step_part=?,           # Lagrange multiplier warm start
            int max_interp=?,           # Number of interps before stepping
            double tol=?,            # Constraint tolerance
            int verbose=?)


cdef void cpdas_lite(const size_t t,
                     const double delta,       # MSE constraint
                     const double *wi,               # Observation weights
                     double *x,                # Noisey Observations Denoise In Place
                     double *z,                # Dual variable warm start
                     double *lambda_,          # Lagrange multiplier warm start
                     int verbose) nogil


cdef double compute_scale(const size_t t, 
                          const double *y, 
                          const double delta) nogil

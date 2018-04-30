# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import numpy as np
cimport numpy as np

from libcpp cimport bool

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
# -------------------- Primal Dual Active Set Wrappers ----------------------- #
# ---------------------------------------------------------------------------- #


cdef cpdas(const double[::1] y,        # Observations
            const double delta,         # MSE constraint
            double[::1] wi=?,           # Observation weights
            double[::1] z_hat=?,        # Dual variable warm start
            double lambda_=?,           # Lagrange multiplier warm start
            int max_interp=?,           # Number of interps before stepping
            double tol=?,               # Constraint tolerance
            int verbose=?)


cdef cps_cpdas(const double[::1] y,        # Observations
                const double delta,         # MSE constraint
                double[::1] wi=?,           # Observation weights
                double[::1] z_hat=?,        # Dual variable warm start
                double lambda_=?,           # Lagrange multiplier warm start
                double tol=?,               # Constraint tolerance
                int verbose=?)


cdef lpdas(double[::1] y,
            const double lambda_,
            double[::1] wi=?,
            double[::1] z_hat=?,
            double p=?,    
            int m=?,
            double delta_s=?,
            double delta_e=?,
            int maxiter=?,
            int verbose=?)


# ---------------------------------------------------------------------------- #
# ------------------------------ ADMM Wrappers ------------------------------- #
# ---------------------------------------------------------------------------- #


cpdef cps_cadmm(const double[::1] y,        # Observations
                const double delta,         # MSE constraint
                double[::1] w=?,        # Observation weights
                double[::1] beta=?,      # Primal Warm Start
                double[::1] alpha=?,     # Dual variable warm start
                double[::1] u=?,     # Dual variable warm start
                double lambda_=?,          # Lagrange multiplier warm start
                double tol=?,            # Constraint tolerance
                int verbose=?)
 

cpdef ladmm(const double[::1] y,        # Observations
            const double lambda_,         # MSE constraint
            double[::1] w=?,        # Observation weights
            double[::1] beta=?,      # Primal Warm Start
            double[::1] alpha=?,     # Dual variable warm start
            double[::1] u=?,     # Dual variable warm start
            int verbose=?)


# ---------------------------------------------------------------------------- #
# --------------------- Interion Point Method Wrappers ----------------------- #
# ---------------------------------------------------------------------------- #


cdef lipm(double[::1] y,
          double lambda_,
          bool max_lambda=?,
          double tol=?,
          int maxiter=?,
          int verbose=?)

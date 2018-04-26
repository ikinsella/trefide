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
 

#cdef extern from "glmgen.h":
#    void tf_admm(double * x, double * y, double * w, int n, int k, int family,
#                 int max_iter, int lam_flag, double * lambda_, 
#                 int nlambda, double lambda_min_ratio, int * df,
#                 double * beta, double * obj, int * iter, int * status, 
#                 double rho, double obj_tol, double obj_tol_newton, double alpha_ls,
#                 double gamma_ls, int max_iter_ls, int max_inner_iter, int verbose) nogil

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

cdef ipm(double[::1] y,
          double lambda_,
          bool max_lambda=?,
          double tol=?,
          int maxiter=?,
          int verbose=?)

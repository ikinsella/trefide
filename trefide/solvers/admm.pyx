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
# ------------------------------ ADMM Wrappers ------------------------------- #
# ---------------------------------------------------------------------------- #

cdef extern from "glmgen.h":
    void tf_admm(double * x, double * y, double * w, int n, int k, int family,
                 int max_iter, int lam_flag, double * lambda_, 
                 int nlambda, double lambda_min_ratio, int * df,
                 double * beta, double * obj, int * iter, int * status, 
                 double rho, double obj_tol, double obj_tol_newton, double alpha_ls,
                 double gamma_ls, int max_iter_ls, int max_inner_iter, int verbose) nogil

cpdef ladmm(double[::1] y):
    """ Handle to weighted pdas solver allowing warm start intialization"""
    cdef int n = y.shape[0]
    cdef int k = 2
    cdef int family = 0
    cdef int max_iter = 500
    cdef int lam_flag = 0
    cdef int nlambda = 50
    cdef double lambda_min_ratio = 1e-5
    cdef double rho = 1
    cdef double obj_tol = 1e-6
    cdef double obj_tol_newton = 1e-6
    cdef double alpha_ls = 0.5
    cdef double gamma_ls = 0.9
    cdef int max_iter_ls = 20
    cdef int max_iter_newton = 50
    cdef int max_iter_outer = max_iter
    cdef int verbose= 0


    # Allocate Space
    cdef double[::1] x = np.arange(n, dtype=np.float64)
    cdef double[::1] w = np.ones(n, dtype=np.float64)
    cdef double[::1] lambda_ = np.zeros(nlambda, np.float64)
    cdef int[::1] df = np.zeros(nlambda, dtype=np.int32)
    cdef double[::1] beta = np.zeros(nlambda * n, dtype=np.float64)
    cdef double[::1] obj = np.zeros(nlambda * max_iter_outer, dtype=np.float64)
    cdef int[::1] iters = np.zeros(nlambda, dtype=np.int32)
    cdef int[::1] status = np.zeros(nlambda, dtype=np.int32)

    with nogil:
        tf_admm(&x[0], &y[0], &w[0], n, k, family, max_iter, lam_flag,
                &lambda_[0], nlambda, lambda_min_ratio, &df[0], &beta[0],
                &obj[0], &iters[0], &status[0], rho, obj_tol, obj_tol_newton,
                alpha_ls, gamma_ls, max_iter_ls, max_iter_newton, verbose);
    
    return beta, lambda_, df, obj, iters, status 

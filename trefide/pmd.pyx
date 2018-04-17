# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False


cdef extern from "/home/ian/devel/trefide/src/pmd/pmd.cpp":
    size_t factor_patch(const size_t d1, 
                        const size_t d2, 
                        const size_t t,
                        double* R, 
                        double* U,
                        double* V,
                        const double lambda_tv,
                        const double spatial_thresh,
                        const size_t max_components,
                        const size_t max_iters,
                        const double tol) nogil

cpdef size_t call_pmd(const size_t d1, 
                      const size_t d2, 
                      const size_t t,
                      double[::1] Y, 
                      double[::1] U,
                      double[::1] V,
                      const double lambda_tv,
                      const double spatial_thresh,
                      const size_t max_components,
                      const size_t max_iters,
                      const double tol) nogil:
    """ 
    Iteratively factor patch into spatial and temporal components 
    with a penalized matrix decomposition
    """ 
    with nogil:
        return factor_patch(d1, d2, t, &Y[0], &U[0], &V[0], lambda_tv, 
                            spatial_thresh, max_components, max_iters, tol)

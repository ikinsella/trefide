# cython: cdivision=True
# cython: boundscheck=True
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import os

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free

# -----------------------------------------------------------------------------#
# ----------------- Sequential Single-Block Processing Funcs ------------------#
# -----------------------------------------------------------------------------#


cdef extern from "mkl.h":
    void cblas_dcopy (const int n,
                      const double *x, 
                      const int incx, 
                      double *y, 
                      const int incy) nogil


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
    return factor_patch(d1, d2, t, &Y[0], &U[0], &V[0], lambda_tv, 
                        spatial_thresh, max_components, max_iters, tol)


# -----------------------------------------------------------------------------#
# --------------- Multi-Threaded Multi-Block Processing Funcs -----------------#
# -----------------------------------------------------------------------------#



cpdef block_pmd(const size_t d1, 
                const size_t d2, 
                const size_t t,
                double[:, :, ::1] Y, 
                const size_t bheight,
                const size_t bwidth,
                const double lambda_tv,
                const double spatial_thresh,
                const size_t max_components,
                const size_t max_iters,
                const double tol):

    print("Computing Block Indices...")
    # Compute block-start indices
    cdef size_t nbi = int(d1/bheight)
    cdef size_t nbj = int(d2/bwidth)
    cdef int num_blocks = nbi * nbj
    indices = np.transpose([np.tile(range(nbi), nbj),
                            np.repeat(range(nbj), nbi)])

    print("Allocating memoryviews...")
    # Preallocate Space For Residuals & Outputs
    cdef double[:, ::1] R = np.empty((num_blocks, bheight*bwidth*t), 
                                     dtype=np.float64)
    cdef double[:,::1] U = np.empty((num_blocks, bheight*bwidth*max_components), 
                                     dtype=np.float64)
    cdef double[:,::1] V = np.empty((num_blocks, t*max_components), dtype=np.float64)
    cdef size_t[::1] K = np.empty((num_blocks,), dtype=np.uint64)

    print("Copying...")
    # Fill Residual Blocks
    cdef size_t i, j, k, bi, bj
    for bi, bj in indices:
        for k in range(t):
            for j in range(bwidth):
                for i in range(bheight):
                    R[bi + (bj * nbi), i + (j * bheight) + (k * bheight * bwidth)] =\
                            Y[(bi * bheight) + i, (bj * bwidth) + j, k]

    print("copy completed")
    cdef int b
    with nogil, parallel(num_threads=2):
        for b in prange(num_blocks, schedule="guided"):
             k = call_pmd(bheight, bwidth, t, 
                            R[b,:], U[b,:], V[b,:],
                            lambda_tv, spatial_thresh, 
                            max_components, max_iters,tol)
            
    # Return Components & Residual
    return R, U, V, K, indices

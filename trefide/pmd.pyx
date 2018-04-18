# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import os

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange
from libc.stdlib cimport abort, calloc, malloc, free

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
    size_t factor_patch(const int d1, 
                        const int d2, 
                        const int t,
                        double* R, 
                        double* U,
                        double* V,
                        const double lambda_tv,
                        const double spatial_thresh,
                        const size_t max_components,
                        const size_t max_iters,
                        const double tol) nogil

    
cdef extern from "/home/ian/devel/trefide/src/pmd/pmd.cpp":
    void parrallel_factor_patch(const int bheight, 
                                const int bwidth, 
                                const int t,
                                const int b,
                                double** Rpt, 
                                double** Upt,
                                double** Vpt,
                                size_t* Kpt,
                                const double lambda_tv,
                                const double spatial_thresh,
                                const size_t max_components,
                                const size_t max_iters,
                                const double tol) nogil


cpdef size_t patch_pmd(const size_t d1, 
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
    #with nogil:
    return factor_patch(d1, d2, t, &Y[0], &U[0], &V[0], lambda_tv, 
                        spatial_thresh, max_components, max_iters, tol)


# -----------------------------------------------------------------------------#
# --------------- Multi-Threaded Multi-Block Processing Funcs -----------------#
# -----------------------------------------------------------------------------#


cpdef blockwise_pmd(const int d1, 
                    const int d2, 
                    const int t,
                    double[:, :, ::1] Y, 
                    const int bheight,
                    const int bwidth,
                    const double lambda_tv,
                    const double spatial_thresh,
                    const size_t max_components,
                    const size_t max_iters,
                    const double tol):

    # Initialize Counters
    cdef size_t iu, ku
    cdef int i, j, k, bi, bj
    cdef int nbi = int(d1/bheight)
    cdef int nbj = int(d2/bwidth)
    cdef int b
    cdef int num_blocks = nbi * nbj

    # Compute block-start indices
    indices = np.transpose([np.tile(range(nbi), nbj),
                            np.repeat(range(nbj), nbi)])

    # Preallocate Space For Residuals & Outputs
    cdef double[:,::1] U = np.empty((num_blocks, bheight * bwidth * max_components),
                                    dtype=np.float64)
    cdef double[:,::1] V = np.empty((num_blocks, t * max_components), 
                                    dtype=np.float64)
    cdef size_t[::1] K = np.empty((num_blocks,), dtype=np.uint64)
    cdef double** Rp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Vp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Up = <double **> malloc(num_blocks * sizeof(double*))
    with nogil:
        for b in range(num_blocks):
            Rp[b] = <double *> malloc(bheight * bwidth * t * sizeof(double))
            Up[b] = <double *> malloc(bheight * bwidth * max_components * sizeof(double))
            Vp[b] = <double *> malloc(t * max_components * sizeof(double))
 
    # Copy Contents Of Raw Blocks Into Residual Pointers
    with nogil:
        # Fill Residual Blocks
        for bj in range(nbj):
            for bi in range(nbi):
                for k in range(t):
                    for j in range(bwidth):
                        for i in range(bheight):
                            Rp[bi + (bj * nbi)][i + (j * bheight) + (k * bheight * bwidth)] =\
                                    Y[(bi * bheight) + i, (bj * bwidth) + j, k]

    # Factor Blocks In Parallel
    with nogil:  #, parallel(num_threads=1): 
        for b in range(num_blocks):#prange(num_blocks, schedule="static"):
             K[b] = factor_patch(bheight, bwidth, t, 
                                 Rp[b], Up[b], Vp[b],
                                 lambda_tv, spatial_thresh, 
                                 max_components, max_iters,tol)

    # Copy Component Contents Back From Pointer To Memoryview
    with nogil: 
        for b in range(num_blocks):
            for iu in range(max_components * bheight * bwidth):
                U[b, iu] = Up[b][iu]
            for ku in range(max_components * t):
                V[b, ku] = Vp[b][ku]

    # Free Memory Allocated To Pointers
    with nogil:
        for b in range(num_blocks):
            free(Rp[b])
            free(Up[b])
            free(Vp[b])
        free(Rp)
        free(Up)
        free(Vp)
            
    return (np.asarray(U).reshape((num_blocks, bheight, bwidth, max_components), order='F'), 
            np.asarray(V).reshape((num_blocks, max_components, t), order='C'), 
            np.asarray(K), 
            indices)

    
cpdef parallel_pmd(const int d1, 
                   const int d2, 
                   const int t,
                   double[:, :, ::1] Y, 
                   const int bheight,
                   const int bwidth,
                   const double lambda_tv,
                   const double spatial_thresh,
                   const size_t max_components,
                   const size_t max_iters,
                   const double tol):

    # Initialize Counters
    cdef size_t iu, ku
    cdef int i, j, k, bi, bj
    cdef int nbi = int(d1/bheight)
    cdef int nbj = int(d2/bwidth)
    cdef int b
    cdef int num_blocks = nbi * nbj

    # Compute block-start indices
    indices = np.transpose([np.tile(range(nbi), nbj),
                            np.repeat(range(nbj), nbi)])

    # Preallocate Space For Residuals & Outputs
    cdef double[:,::1] U = np.empty((num_blocks, bheight * bwidth * max_components),
                                    dtype=np.float64)
    cdef double[:,::1] V = np.empty((num_blocks, t * max_components), 
                                    dtype=np.float64)
    cdef size_t[::1] K = np.empty((num_blocks,), dtype=np.uint64)
    cdef double** Rp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Vp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Up = <double **> malloc(num_blocks * sizeof(double*))
    with nogil:

        # Allocate Memory To Pointers
        for b in range(num_blocks):
            Rp[b] = <double *> malloc(bheight * bwidth * t * sizeof(double))
            Up[b] = <double *> malloc(bheight * bwidth * max_components * sizeof(double))
            Vp[b] = <double *> malloc(t * max_components * sizeof(double))
 
        # Touch Memory To Make Sure It Is Initialized
        for b in range(num_blocks):
            for iu in range(max_components * bheight * bwidth):
                Up[b][iu] = 1
            for ku in range(max_components * t):
                Vp[b][ku] = 1

        # Copy Contents Of Raw Blocks Into Residual Pointers
        for bj in range(nbj):
            for bi in range(nbi):
                for k in range(t):
                    for j in range(bwidth):
                        for i in range(bheight):
                            Rp[bi + (bj * nbi)][i + (j * bheight) + (k * bheight * bwidth)] =\
                                    Y[(bi * bheight) + i, (bj * bwidth) + j, k]

        # Factor Blocks In Parallel
        parrallel_factor_patch(bheight, bwidth, t, num_blocks, 
                               Rp, Up, Vp, &K[0],
                               lambda_tv, spatial_thresh, 
                               max_components, max_iters,tol)

        # Copy Component Contents Back From Pointer To Memoryview
        for b in range(num_blocks):
            for iu in range(max_components * bheight * bwidth):
                U[b, iu] = Up[b][iu]
            for ku in range(max_components * t):
                V[b, ku] = Vp[b][ku]

        # Free Memory Allocated To Pointers
        for b in range(num_blocks):
            free(Rp[b])
            free(Up[b])
            free(Vp[b])
        free(Rp)
        free(Up)
        free(Vp)
            
    return (np.asarray(U).reshape((num_blocks, bheight, bwidth, max_components), order='F'), 
            np.asarray(V).reshape((num_blocks, max_components, t), order='C'), 
            np.asarray(K), 
            indices)

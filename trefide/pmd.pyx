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
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":
    size_t pmd(const int d1, 
               const int d2, 
               const int t,
               double* R, 
               double* U,
               double* V,
               const double lambda_tv,
               const double spatial_thresh,
               const size_t max_components,
               const size_t consec_failures,
               const size_t max_iters,
               const double tol) nogil

    void batch_pmd(const int bheight, 
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
                   const size_t consec_failures,
                   const size_t max_iters,
                   const double tol) nogil


# -----------------------------------------------------------------------------#
# -------------------------- Single-Block Wrapper -----------------------------#
# -----------------------------------------------------------------------------#


cpdef size_t decompose(const int d1, 
                       const int d2, 
                       const int t,
                       double[::1] Y, 
                       double[::1] U,
                       double[::1] V,
                       const double lambda_tv,
                       const double spatial_thresh,
                       const size_t max_components,
                       const size_t consec_failures,
                       const size_t max_iters,
                       const double tol) nogil:
    """ Wrap the single patch cpp PMD functions """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        return pmd(d1, d2, t, &Y[0], &U[0], &V[0], lambda_tv, 
                   spatial_thresh, max_components, consec_failures, 
                   max_iters, tol)


# -----------------------------------------------------------------------------#
# --------------------------- Multi-Block Wrappers ----------------------------#
# -----------------------------------------------------------------------------#


cpdef batch_decompose(const int d1, 
                      const int d2, 
                      const int t,
                      double[:, :, ::1] Y, 
                      const int bheight,
                      const int bwidth,
                      const double lambda_tv,
                      const double spatial_thresh,
                      const size_t max_components,
                      const size_t consec_failures,
                      const size_t max_iters,
                      const double tol):
    """ Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function 
     factor_patch with OpenMP directives to parallelize batch processing."""

    # Assert Evenly Divisible FOV/Block Dimensions
    assert d1 % bheight == 0 , "Input FOV height must be an evenly divisible by block height."
    assert d2 % bwidth == 0 , "Input FOV width must be evenly divisible by block width."

    # Initialize Counters
    cdef size_t iu, ku
    cdef int i, j, k, b, bi, bj
    cdef int nbi = int(d1/bheight)
    cdef int nbj = int(d2/bwidth)
    cdef int num_blocks = nbi * nbj

    # Compute block-start indices and spatial cutoff
    indices = np.transpose([np.tile(range(nbi), nbj), np.repeat(range(nbj), nbi)])

    # Preallocate Space For Outputs
    cdef double[:,::1] U = np.zeros((num_blocks, bheight * bwidth * max_components), dtype=np.float64)
    cdef double[:,::1] V = np.zeros((num_blocks, t * max_components), dtype=np.float64)
    cdef size_t[::1] K = np.empty((num_blocks,), dtype=np.uint64)

    # Allocate Input Pointers
    cdef double** Rp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Vp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Up = <double **> malloc(num_blocks * sizeof(double*))

    # Release Gil Prior To Referencing Address & Calling Multithreaded Code
    with nogil:

        # Assign Pre-allocated Output Memory To Pointer Array & Allocate Residual Pointers
        for b in range(num_blocks):
            Rp[b] = <double *> malloc(bheight * bwidth * t * sizeof(double))
            Up[b] = &U[b,0]
            Vp[b] = &V[b,0] 

        # Copy Contents Of Raw Blocks Into Residual Pointers
        for bj in range(nbj):
            for bi in range(nbi):
                for k in range(t):
                    for j in range(bwidth):
                        for i in range(bheight):
                            Rp[bi + (bj * nbi)][i + (j * bheight) + (k * bheight * bwidth)] =\
                                    Y[(bi * bheight) + i, (bj * bwidth) + j, k]

        # Factor Blocks In Parallel
        batch_pmd(bheight, bwidth, t, num_blocks, Rp, Up, Vp, &K[0],
                  lambda_tv, spatial_thresh, max_components, consec_failures, 
                  max_iters, tol)

        # Free Allocated Memory
        for b in range(num_blocks):
            free(Rp[b])
        free(Rp)
        free(Up)
        free(Vp)
            
    # Format Components & Return To Numpy Array
    return (np.asarray(U).reshape((num_blocks, bheight, bwidth, max_components), order='F'), 
            np.asarray(V).reshape((num_blocks, max_components, t), order='C'), 
            np.asarray(K), indices.astype(np.uint64))


cpdef double[:,:,::1] batch_recompose(double[:, :, :, :] U, 
                                      double[:,:,::1] V, 
                                      size_t[::1] K, 
                                      size_t[:,:] indices):
    """ Reconstruct A Denoised Movie """

    # Get Block Size Info From Spatial
    cdef size_t num_blocks = U.shape[0]
    cdef size_t bheight = U.shape[1]
    cdef size_t bwidth = U.shape[2]
    cdef size_t t = V.shape[2]

    # Get Mvie Size Infro From Indices
    cdef size_t nbi, nbj
    nbi = np.max(indices[:,0]) + 1
    nbj = np.max(indices[:,1]) + 1
    cdef size_t d1 = nbi * bheight
    cdef size_t d2 = nbj * bwidth

    # Allocate Space For reconstructed Movies
    Yd = np.zeros(d1*d2*t, dtype=np.float64).reshape((d1,d2,t))

    # Loop Over Blocks
    cdef size_t bdx, idx, jdx, kdx
    for bdx in range(nbi*nbj):
        idx = indices[bdx,0] * bheight
        jdx = indices[bdx,1] * bwidth
        Yd[idx:idx+bheight, jdx:jdx+bwidth,:] += np.reshape(
                np.dot(U[bdx,:,:,:K[bdx]],
                       V[bdx,:K[bdx],:]),
                (bheight,bwidth,t),
                order='F')
    # Rank One updates
    return np.asarray(Yd)



# -----------------------------------------------------------------------------#
# --------------------------- Overlapping Wrappers ----------------------------#
# -----------------------------------------------------------------------------#


cpdef double[:,:,::1] weighted_batch_recompose(double[:, :, :, :] U, 
                                               double[:,:,:] V, 
                                               size_t[:] K, 
                                               size_t[:,:] indices,
                                               double[:,:] W):
    """ Reconstruct A Denoised Movie """

    # Get Block Size Info From Spatial
    cdef size_t num_blocks = U.shape[0]
    cdef size_t bheight = U.shape[1]
    cdef size_t bwidth = U.shape[2]
    cdef size_t t = V.shape[2]

    # Get Mvie Size Infro From Indices
    cdef size_t nbi, nbj
    nbi = len(np.unique(indices[:,0]))
    idx_offset = np.min(indices[:,0])
    nbj = len(np.unique(indices[:,1]))
    jdx_offset = np.min(indices[:,1])
    cdef size_t d1 = nbi * bheight
    cdef size_t d2 = nbj * bwidth

    # Allocate Space For reconstructed Movies
    Yd = np.zeros(d1*d2*t, dtype=np.float64).reshape((d1,d2,t))

    # Loop Over Blocks
    cdef size_t bdx, idx, jdx, kdx
    for bdx in range(nbi*nbj):
        idx = (indices[bdx,0] - idx_offset) * bheight
        jdx = (indices[bdx,1] - jdx_offset) * bwidth
        Yd[idx:idx+bheight, jdx:jdx+bwidth,:] += np.reshape(
                np.dot(U[bdx,:,:,:K[bdx]],
                       V[bdx,:K[bdx],:]),
                (bheight,bwidth,t),
                order='F') * np.asarray(W[:,:,None]) 
    return np.asarray(Yd)


cpdef overlapping_batch_denoise(const int d1, 
                                const int d2, 
                                const int t,
                                double[:, :, ::1] Y, 
                                const int bheight,
                                const int bwidth,
                                const double lambda_tv,
                                const double spatial_thresh,
                                const size_t max_components,
                                const size_t consec_failures,
                                const size_t max_iters,
                                const double tol):
    """ 4x batch denoiser """

    # Assert Even Blockdims
    assert bheight % 2 == 0 , "Block height must be an even integer."
    assert bwidth % 2 == 0 , "Block width must be an even integer."
    
    # Assert Even Blockdims
    assert d1 % bheight == 0 , "Input FOV height must be an evenly divisible by block height."
    assert d2 % bwidth == 0 , "Input FOV width must be evenly divisible by block width."
    
    # Declare internal vars
    cdef int i,j
    cdef int hbheight = bheight/2
    cdef int hbwidth = bwidth/2
    cdef int nbrow = d1/bheight
    cdef int nbcol = d2/bwidth

    # -------------------- Construct Combination Weights ----------------------#
    
    # Generate Single Quadrant Weighting matrix
    cdef double[:,:] ul_weights = np.empty((hbheight, hbwidth), dtype=np.float64)
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = min(i, j)

    # Compute Cumulative Overlapped Weights (Normalizing Factor)
    cdef double[:,:] cum_weights = np.asarray(ul_weights) +\
            np.fliplr(ul_weights) + np.flipud(ul_weights) +\
            np.fliplr(np.flipud(ul_weights)) 

    # Normalize By Cumulative Weights
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = ul_weights[i,j] / cum_weights[i,j]


    # Construct Full Weighting Matrix From Normalize Quadrant
    cdef double[:,:] W = np.hstack([np.vstack([ul_weights, 
                                               np.flipud(ul_weights)]),
                                    np.vstack([np.fliplr(ul_weights), 
                                               np.fliplr(np.flipud(ul_weights))])]) 

    # ---------------- Handle Blocks Overlays One At A Time --------------#
    Yd = np.zeros((d1,d2,t), dtype=np.float64)

    # ----------- Original Overlay
    # Only Need To Process Full-Size Blocks
    U, V, K, I = batch_decompose(d1, d2, t, Y, bheight, bwidth, lambda_tv,
                                 spatial_thresh, max_components, 
                                 consec_failures, max_iters, tol)
    Yd += weighted_batch_recompose(U, V, K, I, W)
 
    # ---------- Add Vertical Skew Block Overlay To Reconstruction
    # Full Blocks
    U, V, K, I = batch_decompose(d1 - bheight, d2, t, 
                                Y[hbheight:d1-hbheight,:,:], 
                                bheight, bwidth, 
                                lambda_tv, spatial_thresh, max_components, 
                                consec_failures, max_iters, tol)
    Yd[hbheight:d1-hbheight,:,:] += weighted_batch_recompose(U, V, K, I, W)

    # wide half blocks
    U, V, K, I = batch_decompose(bheight, d2, t, 
                                 np.vstack([Y[:hbheight,:,:], Y[d1-hbheight:,:,:]]),
                                 hbheight, bwidth, 
                                 lambda_tv, spatial_thresh, max_components, 
                                 consec_failures, max_iters, tol)
    Yd[:hbheight,:,:] += weighted_batch_recompose(U[::2], V[::2], K[::2], I[::2],  W[hbheight:, :])
    Yd[d1-hbheight:,:,:] += weighted_batch_recompose(U[1::2], V[1::2], K[1::2], I[1::2], W[:hbheight, :])
    
    # --------------Horizontal Skew
    # Full Blocks
    U, V, K, I = batch_decompose(d1, d2 - bwidth, t, 
                                 Y[:, hbwidth:d2-hbwidth,:], 
                                 bheight, bwidth, 
                                 lambda_tv, spatial_thresh, max_components, 
                                 consec_failures, max_iters, tol)
    Yd[:, hbwidth:d2-hbwidth,:] += weighted_batch_recompose(U, V, K, I, W)

    # tall half blocks
    U, V, K, I = batch_decompose(d1, bwidth, t, 
                                 np.hstack([Y[:,:hbwidth,:], Y[:,d2-hbwidth:,:]]),
                                 bheight, hbwidth, 
                                 lambda_tv, spatial_thresh, max_components, 
                                 consec_failures, max_iters, tol)
    Yd[:,:hbwidth,:] += np.asarray(weighted_batch_recompose(U[:nbrow], V[:nbrow], K[:nbrow], I[:nbrow],  W[:, hbwidth:]))
    Yd[:,d2-hbwidth:,:] += weighted_batch_recompose(U[nbrow:], V[nbrow:], K[nbrow:], I[nbrow:], W[:, :hbwidth])

    # -------------Diagonal Skew
    # Full Blocks
    U, V, K, I = batch_decompose(d1 - bheight, d2 - bwidth, t, 
                                 Y[hbheight:d1-hbheight, hbwidth:d2-hbwidth, :], 
                                  bheight, bwidth, 
                                  lambda_tv, spatial_thresh, max_components, 
                                  consec_failures, max_iters, tol)
    Yd[hbheight:d1-hbheight, hbwidth:d2-hbwidth, :] += weighted_batch_recompose(U, V, K, I, W)

    # tall half blocks
    U, V, K, I = batch_decompose(d1 - bheight, bwidth, t, 
                                 np.hstack([Y[hbheight:d1-hbheight, :hbwidth, :],
                                            Y[hbheight:d1-hbheight, d2-hbwidth:, :]]),
                                 bheight, hbwidth, 
                                 lambda_tv, spatial_thresh, max_components, 
                                 consec_failures, max_iters, tol)
    Yd[hbheight:d1-hbheight,:hbwidth,:] += weighted_batch_recompose(U[:nbrow-1], V[:nbrow-1], K[:nbrow-1], I[:nbrow-1],  W[:, hbwidth:])
    Yd[hbheight:d1-hbheight,d2-hbwidth:,:] += weighted_batch_recompose(U[nbrow-1:], V[nbrow-1:], K[nbrow-1:], I[nbrow-1:], W[:, :hbwidth])

    # wide half blocks
    U, V, K, I = batch_decompose(bheight, d2 - bwidth, t, 
                                 np.vstack([Y[:hbheight, hbwidth:d2-hbwidth, :], 
                                            Y[d1-hbheight:, hbwidth:d2-hbwidth, :]]),
                                 hbheight, bwidth, 
                                 lambda_tv, spatial_thresh, max_components, 
                                 consec_failures, max_iters, tol) 
    Yd[:hbheight,hbwidth:d2-hbwidth,:] += weighted_batch_recompose(U[::2], V[::2], K[::2], I[::2],  W[hbheight:, :])
    Yd[d1-hbheight:,hbwidth:d2-hbwidth,:] += weighted_batch_recompose(U[1::2], V[1::2], K[1::2], I[1::2], W[:hbheight, :])

    # Corners
    U, V, K, I = batch_decompose(bheight, bwidth, t, 
                                 np.hstack([np.vstack([Y[:hbheight, :hbwidth, :], 
                                                       Y[d1-hbheight:, :hbwidth, :]]),
                                            np.vstack([Y[:hbheight, d2-hbwidth:, :], 
                                                       Y[d1-hbheight:, d2-hbwidth:, :]])]),
                                 hbheight, hbwidth, 
                                 lambda_tv, spatial_thresh, max_components, 
                                 consec_failures, max_iters, tol)
    Yd[:hbheight,:hbwidth,:] += weighted_batch_recompose(U[:1], V[:1], K[:1], I[:1],  W[hbheight:, hbwidth:])
    Yd[d1-hbheight:,:hbwidth,:] += weighted_batch_recompose(U[1:2], V[1:2], K[1:2], I[1:2],  W[:hbheight, hbwidth:])
    Yd[:hbheight,d2-hbwidth:,:] += weighted_batch_recompose(U[2:3], V[2:3], K[2:3], I[2:3],  W[hbheight:, :hbwidth])
    Yd[d1-hbheight:,d2-hbwidth:,:] += weighted_batch_recompose(U[3:], V[3:], K[3:], I[3:],  W[:hbheight:, :hbwidth])
    return np.asarray(Yd)

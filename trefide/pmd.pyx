# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import os
import time

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange
from libc.stdlib cimport abort, calloc, malloc, free


FOV_BHEIGHT_WARNING = "Input FOV height must be an evenly divisible by block height."
FOV_BWIDTH_WARNING = "Input FOV width must be evenly divisible by block width." 
DSUB_BHEIGHT_WARNING = "Block height must be evenly divisible by spatial downsampling factor."
DSUB_BWIDTH_WARNING = "Block width must be evenly divisible by spatial downsampling factor."
TSUB_FRAMES_WARNING = "Num Frames must be evenly divisible by temporal downsampling factor."


# -----------------------------------------------------------------------------#
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":
    
    size_t pmd(const int d1, 
               const int d2, 
               int d_sub, 
               const int t,
               int t_sub,
               double* R, 
               double* R_ds,
               double* U,
               double* V,
               const double spatial_thresh,
               const double temporal_thresh,
               const size_t max_components,
               const size_t consec_failures,
               const int max_iters_main,
               const int max_iters_init,
               const double tol) nogil

    void batch_pmd(const int bheight,
                   const int bwidth, 
                   int d_sub,
                   const int t,
                   int t_sub,
                   const int b,
                   double** Rpt, 
                   double** Rpt_ds, 
                   double** Upt,
                   double** Vpt,
                   size_t* Kpt,
                   const double spatial_thresh,
                   const double temporal_thresh,
                   const size_t max_components,
                   const size_t consec_failures,
                   const size_t max_iters_main,
                   const size_t max_iters_init,
                   const double tol) nogil

    void downsample_3d(const int d1, 
                       const int d2, 
                       const int d_sub, 
                       const int t, 
                       const int t_sub, 
                       const double *Y, 
                       double *Y_ds) nogil

# -----------------------------------------------------------------------------#
# -------------------------- Single-Block Wrapper -----------------------------#
# -----------------------------------------------------------------------------#


cpdef size_t decompose(const int d1, 
                       const int d2, 
                       const int t,
                       double[::1] Y, 
                       double[::1] U,
                       double[::1] V,
                       const double spatial_thresh,
                       const double temporal_thresh,
                       const size_t max_components,
                       const size_t consec_failures,
                       const size_t max_iters_main,
                       const size_t max_iters_init,
                       const double tol) nogil:
    """ Wrap the single patch cpp PMD functions """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        return pmd(d1, d2, 1, t, 1, &Y[0], NULL, &U[0], &V[0], 
                   spatial_thresh, temporal_thresh,
                   max_components, consec_failures, 
                   max_iters_main, max_iters_init, tol)


cpdef size_t decimated_decompose(const int d1, 
                                 const int d2, 
                                 int d_sub,
                                 const int t,
                                 int t_sub,
                                 double[::1] Y, 
                                 double[::1] Y_ds, 
                                 double[::1] U,
                                 double[::1] V,
                                 const double spatial_thresh,
                                 const double temporal_thresh,
                                 const size_t max_components,
                                 const size_t consec_failures,
                                 const int max_iters_main,
                                 const int max_iters_init,
                                 const double tol) nogil:
    """ Wrap the single patch cpp PMD functions """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        return pmd(d1, d2, d_sub, t, t_sub, &Y[0], &Y_ds[0], &U[0], &V[0], 
                   spatial_thresh, temporal_thresh,
                   max_components, consec_failures, 
                   max_iters_main, max_iters_init, tol);

# -----------------------------------------------------------------------------#
# --------------------------- Multi-Block Wrappers ----------------------------#
# -----------------------------------------------------------------------------#


cpdef batch_decompose(const int d1, 
                      const int d2, 
                      const int t,
                      double[:, :, ::1] Y, 
                      const int bheight,
                      const int bwidth,
                      const double spatial_thresh,
                      const double temporal_thresh,
                      const size_t max_components,
                      const size_t consec_failures,
                      const size_t max_iters_main,
                      const size_t max_iters_init,
                      const double tol,
                      int d_sub = 1,
                      int t_sub = 1):
    """ Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function 
     factor_patch with OpenMP directives to parallelize batch processing."""

    # Assert Evenly Divisible FOV/Block Dimensions
    assert d1 % bheight == 0 , FOV_BHEIGHT_WARNING
    assert d2 % bwidth == 0 , FOV_BWIDTH_WARNING
    assert bheight % d_sub == 0 , DSUB_BHEIGHT_WARNING
    assert bwidth % d_sub == 0 , DSUB_BWIDTH_WARNING
    assert t % t_sub == 0 , TSUB_FRAMES_WARNING    
    
    # Initialize Counters
    cdef size_t iu, ku
    cdef int i, j, k, b, bi, bj
    cdef int nbi = int(d1/bheight)
    cdef int nbj = int(d2/bwidth)
    cdef int num_blocks = nbi * nbj
    cdef int bheight_ds = bheight / d_sub
    cdef int bwidth_ds = bwidth / d_sub
    cdef int t_ds = t / t_sub

    # Compute block-start indices and spatial cutoff
    indices = np.transpose([np.tile(range(nbi), nbj), np.repeat(range(nbj), nbi)])

    # Preallocate Space For Outputs
    cdef double[:,::1] U = np.zeros((num_blocks, bheight * bwidth * max_components), dtype=np.float64)
    cdef double[:,::1] V = np.zeros((num_blocks, t * max_components), dtype=np.float64)
    cdef size_t[::1] K = np.empty((num_blocks,), dtype=np.uint64)

    # Allocate Input Pointers
    cdef double** Rp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Rp_ds = <double **> malloc(num_blocks * sizeof(double*))
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
        
        # Decimate Raw Blocks
        if t_sub > 1 or d_sub > 1: 
            for b in range(num_blocks):
                Rp_ds[b] = <double *> malloc((bheight / d_sub) * (bwidth / d_sub) * (t / t_sub) * sizeof(double))
            for b in prange(num_blocks, schedule='guided'):
                downsample_3d(bheight, bwidth, d_sub, t, t_sub, Rp[b], Rp_ds[b]) 
        else:
            for b in range(num_blocks):
                Rp_ds[b] = NULL

        # Factor Blocks In Parallel
        batch_pmd(bheight, bwidth, d_sub, t, t_sub, num_blocks, 
                  Rp, Rp_ds, Up, Vp, &K[0], 
                  spatial_thresh, temporal_thresh,
                  max_components, consec_failures, 
                  max_iters_main, max_iters_init, tol)
        
        # Free Allocated Memory
        for b in range(num_blocks):
            free(Rp[b])
        if t_sub > 1 or d_sub > 1: 
            for b in range(num_blocks):
                free(Rp_ds[b])
        free(Rp)
        free(Rp_ds)
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


cpdef double[:,:,::1] weighted_recompose(double[:, :, :, :] U, 
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


cpdef overlapping_batch_decompose(const int d1, 
                                  const int d2, 
                                  const int t,
                                  double[:, :, ::1] Y, 
                                  const int bheight,
                                  const int bwidth,
                                  const double spatial_thresh,
                                  const double temporal_thresh,
                                  const size_t max_components,
                                  const size_t consec_failures,
                                  const size_t max_iters_main,
                                  const size_t max_iters_init,
                                  const double tol,
                                  int d_sub=1,
                                  int t_sub=1):
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

    # Initialize Outputs
    cdef dict U = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict V = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict K = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    cdef dict I = {'no_skew':{}, 'vert_skew':{}, 'horz_skew':{}, 'diag_skew':{}}
    
    # ---------------- Handle Blocks Overlays One At A Time --------------#

    # ----------- Original Overlay -----------
    # Only Need To Process Full-Size Blocks
    U['no_skew']['full'],\
    V['no_skew']['full'],\
    K['no_skew']['full'],\
    I['no_skew']['full'] = batch_decompose(d1, d2, t, Y, bheight, bwidth,
                                           spatial_thresh,temporal_thresh,
                                           max_components, consec_failures, 
                                           max_iters_main, max_iters_init, 
                                           tol, d_sub=d_sub, t_sub=t_sub)
 
    # ---------- Vertical Skew -----------
    # Full Blocks
    U['vert_skew']['full'],\
    V['vert_skew']['full'],\
    K['vert_skew']['full'],\
    I['vert_skew']['full'] = batch_decompose(d1 - bheight, d2, t, 
                                             Y[hbheight:d1-hbheight,:,:], 
                                             bheight, bwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init,
                                             tol, d_sub=d_sub, t_sub=t_sub)

    # wide half blocks
    U['vert_skew']['half'],\
    V['vert_skew']['half'],\
    K['vert_skew']['half'],\
    I['vert_skew']['half'] = batch_decompose(bheight, d2, t, 
                                             np.vstack([Y[:hbheight,:,:], 
                                                        Y[d1-hbheight:,:,:]]),
                                             hbheight, bwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures, 
                                             max_iters_main, max_iters_init,
                                             tol, d_sub=d_sub, t_sub=t_sub)
    
    # --------------Horizontal Skew---------- 
    # Full Blocks
    U['horz_skew']['full'],\
    V['horz_skew']['full'],\
    K['horz_skew']['full'],\
    I['horz_skew']['full'] = batch_decompose(d1, d2 - bwidth, t, 
                                             Y[:, hbwidth:d2-hbwidth,:], 
                                             bheight, bwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures, 
                                             max_iters_main, max_iters_init,
                                             tol, d_sub=d_sub, t_sub=t_sub)

    # tall half blocks
    U['horz_skew']['half'],\
    V['horz_skew']['half'],\
    K['horz_skew']['half'],\
    I['horz_skew']['half'] = batch_decompose(d1, bwidth, t, 
                                             np.hstack([Y[:,:hbwidth,:],
                                                        Y[:,d2-hbwidth:,:]]),
                                             bheight, hbwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures, 
                                             max_iters_main, max_iters_init, 
                                             tol, d_sub=d_sub, t_sub=t_sub)

    # -------------Diagonal Skew---------- 
    # Full Blocks
    U['diag_skew']['full'],\
    V['diag_skew']['full'],\
    K['diag_skew']['full'],\
    I['diag_skew']['full'] = batch_decompose(d1 - bheight, d2 - bwidth, t, 
                                             Y[hbheight:d1-hbheight,
                                               hbwidth:d2-hbwidth, :], 
                                             bheight, bwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures, 
                                             max_iters_main, max_iters_init, 
                                             tol, d_sub=d_sub, t_sub=t_sub)

    # tall half blocks
    U['diag_skew']['thalf'],\
    V['diag_skew']['thalf'],\
    K['diag_skew']['thalf'],\
    I['diag_skew']['thalf'] = batch_decompose(d1 - bheight, bwidth, t, 
                                             np.hstack([Y[hbheight:d1-hbheight,
                                                          :hbwidth, :],
                                                        Y[hbheight:d1-hbheight,
                                                          d2-hbwidth:, :]]),
                                             bheight, hbwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures, 
                                             max_iters_main, max_iters_init, 
                                             tol, d_sub=d_sub, t_sub=t_sub)

    # wide half blocks
    U['diag_skew']['whalf'],\
    V['diag_skew']['whalf'],\
    K['diag_skew']['whalf'],\
    I['diag_skew']['whalf'] = batch_decompose(bheight, d2 - bwidth, t, 
                                             np.vstack([Y[:hbheight, 
                                                          hbwidth:d2-hbwidth,
                                                          :], 
                                                        Y[d1-hbheight:,
                                                          hbwidth:d2-hbwidth,
                                                          :]]),
                                             hbheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures,
                                             max_iters_main, max_iters_init, 
                                             tol, d_sub=d_sub, t_sub=t_sub) 

    # Corners
    U['diag_skew']['quarter'],\
    V['diag_skew']['quarter'],\
    K['diag_skew']['quarter'],\
    I['diag_skew']['quarter'] = batch_decompose(bheight, bwidth, t, 
                                                np.hstack([
                                                    np.vstack([Y[:hbheight,
                                                                 :hbwidth,
                                                                 :], 
                                                               Y[d1-hbheight:,
                                                                 :hbwidth,
                                                                 :]]),
                                                    np.vstack([Y[:hbheight,
                                                                 d2-hbwidth:,
                                                                 :], 
                                                               Y[d1-hbheight:,
                                                                 d2-hbwidth:,
                                                                 :]])
                                                               ]),
                                                hbheight, hbwidth,
                                                spatial_thresh, temporal_thresh,
                                                max_components, consec_failures, 
                                                max_iters_main, max_iters_init, 
                                                tol, d_sub=d_sub, t_sub=t_sub)

    # Return Weighting Matrix For Reconstruction
    return U, V, K, I, W



cpdef overlapping_batch_recompose(const int d1,
                                  const int d2, 
                                  const int t,
                                  const int bheight,
                                  const int bwidth,
                                  U, V, K, I, W):
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

    # Allocate Space For reconstructed Movies
    Yd = np.zeros((d1,d2,t), dtype=np.float64)

    # ---------------- Handle Blocks Overlays One At A Time --------------#

    # ----------- Original Overlay --------------
    # Only Need To Process Full-Size Blocks
    Yd += weighted_recompose(U['no_skew']['full'],
                             V['no_skew']['full'], 
                             K['no_skew']['full'], 
                             I['no_skew']['full'], 
                             W)
 
    # ---------- Vertical Skew --------------
    # Full Blocks
    Yd[hbheight:d1-hbheight,:,:] += weighted_recompose(U['vert_skew']['full'],
                                                       V['vert_skew']['full'],
                                                       K['vert_skew']['full'], 
                                                       I['vert_skew']['full'], 
                                                       W)
    # wide half blocks
    Yd[:hbheight,:,:] += weighted_recompose(U['vert_skew']['half'][::2],
                                            V['vert_skew']['half'][::2],
                                            K['vert_skew']['half'][::2],
                                            I['vert_skew']['half'][::2],
                                            W[hbheight:, :])
    Yd[d1-hbheight:,:,:] += weighted_recompose(U['vert_skew']['half'][1::2],
                                               V['vert_skew']['half'][1::2],
                                               K['vert_skew']['half'][1::2],
                                               I['vert_skew']['half'][1::2],
                                               W[:hbheight, :])
    
    # --------------Horizontal Skew--------------
    # Full Blocks
    Yd[:, hbwidth:d2-hbwidth,:] += weighted_recompose(U['horz_skew']['full'],
                                                      V['horz_skew']['full'],
                                                      K['horz_skew']['full'],
                                                      I['horz_skew']['full'], 
                                                      W)
    # tall half blocks
    Yd[:,:hbwidth,:] += weighted_recompose(U['horz_skew']['half'][:nbrow],
                                           V['horz_skew']['half'][:nbrow], 
                                           K['horz_skew']['half'][:nbrow], 
                                           I['horz_skew']['half'][:nbrow],  
                                           W[:, hbwidth:])
    Yd[:,d2-hbwidth:,:] += weighted_recompose(U['horz_skew']['half'][nbrow:],
                                              V['horz_skew']['half'][nbrow:],
                                              K['horz_skew']['half'][nbrow:],
                                              I['horz_skew']['half'][nbrow:],
                                              W[:, :hbwidth])

    # -------------Diagonal Skew--------------
    # Full Blocks
    Yd[hbheight:d1-hbheight, hbwidth:d2-hbwidth, :] += weighted_recompose(U['diag_skew']['full'],
                                                                          V['diag_skew']['full'],
                                                                          K['diag_skew']['full'],
                                                                          I['diag_skew']['full'],
                                                                          W)
    # tall half blocks
    Yd[hbheight:d1-hbheight,:hbwidth,:] += weighted_recompose(U['diag_skew']['thalf'][:nbrow-1],
                                                              V['diag_skew']['thalf'][:nbrow-1], 
                                                              K['diag_skew']['thalf'][:nbrow-1], 
                                                              I['diag_skew']['thalf'][:nbrow-1],  
                                                              W[:, hbwidth:])
    Yd[hbheight:d1-hbheight,d2-hbwidth:,:] += weighted_recompose(U['diag_skew']['thalf'][nbrow-1:], 
                                                                 V['diag_skew']['thalf'][nbrow-1:], 
                                                                 K['diag_skew']['thalf'][nbrow-1:], 
                                                                 I['diag_skew']['thalf'][nbrow-1:], 
                                                                 W[:, :hbwidth])
    # wide half blocks
    Yd[:hbheight,hbwidth:d2-hbwidth,:] += weighted_recompose(U['diag_skew']['whalf'][::2], 
                                                             V['diag_skew']['whalf'][::2], 
                                                             K['diag_skew']['whalf'][::2], 
                                                             I['diag_skew']['whalf'][::2],  
                                                             W[hbheight:, :])
    Yd[d1-hbheight:,hbwidth:d2-hbwidth,:] += weighted_recompose(U['diag_skew']['whalf'][1::2], 
                                                                V['diag_skew']['whalf'][1::2], 
                                                                K['diag_skew']['whalf'][1::2], 
                                                                I['diag_skew']['whalf'][1::2], 
                                                                W[:hbheight, :])
    # Corners
    Yd[:hbheight,:hbwidth,:] += weighted_recompose(U['diag_skew']['quarter'][:1],
                                                   V['diag_skew']['quarter'][:1],
                                                   K['diag_skew']['quarter'][:1],
                                                   I['diag_skew']['quarter'][:1],
                                                   W[hbheight:, hbwidth:])
    Yd[d1-hbheight:,:hbwidth,:] += weighted_recompose(U['diag_skew']['quarter'][1:2],
                                                      V['diag_skew']['quarter'][1:2],
                                                      K['diag_skew']['quarter'][1:2],
                                                      I['diag_skew']['quarter'][1:2],
                                                      W[:hbheight, hbwidth:])
    Yd[:hbheight,d2-hbwidth:,:] += weighted_recompose(U['diag_skew']['quarter'][2:3], 
                                                      V['diag_skew']['quarter'][2:3], 
                                                      K['diag_skew']['quarter'][2:3], 
                                                      I['diag_skew']['quarter'][2:3],  
                                                      W[hbheight:, :hbwidth])
    Yd[d1-hbheight:,d2-hbwidth:,:] += weighted_recompose(U['diag_skew']['quarter'][3:], 
                                                         V['diag_skew']['quarter'][3:], 
                                                         K['diag_skew']['quarter'][3:], 
                                                         I['diag_skew']['quarter'][3:],  
                                                         W[:hbheight:, :hbwidth])
    return np.asarray(Yd)

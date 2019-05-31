# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import os
import time
import multiprocessing

import numpy as np
cimport numpy as np

from cython.parallel import parallel, prange
from libc.stdlib cimport abort, calloc, malloc, free
from sklearn.utils.extmath import randomized_svd as svd
from functools import partial
import matplotlib.pyplot as plt
from libcpp cimport bool

FOV_BHEIGHT_WARNING = "Input FOV height must be an evenly divisible by block height."
FOV_BWIDTH_WARNING = "Input FOV width must be evenly divisible by block width." 
DSUB_BHEIGHT_WARNING = "Block height must be evenly divisible by spatial downsampling factor."
DSUB_BWIDTH_WARNING = "Block width must be evenly divisible by spatial downsampling factor."
TSUB_FRAMES_WARNING = "Num Frames must be evenly divisible by temporal downsampling factor."


# -----------------------------------------------------------------------------#
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":

    cdef cppclass PMD_params:
        PMD_params(
            const int _bheight,
            const int _bwidth,
            int _d_sub,
            const int _t,
            int _t_sub,
            const double _spatial_thresh,
            const double _temporal_thresh,
            const size_t _max_components,
            const size_t _consec_failures,
            const size_t _max_iters_main,
            const size_t _max_iters_init,
            const double _tol,
            void *_FFT,
            bool _enable_temporal_denoiser,
            bool _enable_spatial_denoiser) nogil
    
    # size_t pmd(const int d1,
    #            const int d2,
    #            int d_sub,
    #            const int t,
    #            int t_sub,
    #            double* R,
    #            double* R_ds,
    #            double* U,
    #            double* V,
    #            const double spatial_thresh,
    #            const double temporal_thresh,
    #            const size_t max_components,
    #            const size_t consec_failures,
    #            const int max_iters_main,
    #            const int max_iters_init,
    #            const double tol) nogil

    size_t pmd(
            double* R,
            double* R_ds,
            double* U,
            double* V,
            PMD_params *pars) nogil

    # void batch_pmd(const int bheight,
    #                const int bwidth,
    #                int d_sub,
    #                const int t,
    #                int t_sub,
    #                const int b,
    #                double** Rpt,
    #                double** Rpt_ds,
    #                double** Upt,
    #                double** Vpt,
    #                size_t* Kpt,
    #                const double spatial_thresh,
    #                const double temporal_thresh,
    #                const size_t max_components,
    #                const size_t consec_failures,
    #                const size_t max_iters_main,
    #                const size_t max_iters_init,
    #                const double tol) nogil

    void batch_pmd(
               double** Rpt,
               double** Rpt_ds,
               double** Upt,
               double** Vpt,
               size_t* Kpt,
               const int b,
               PMD_params *pars) nogil

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
        # return pmd(d1, d2, 1, t, 1, &Y[0], NULL, &U[0], &V[0],
        #            spatial_thresh, temporal_thresh,
        #            max_components, consec_failures,
        #            max_iters_main, max_iters_init, tol)

        parms = new PMD_params(d1, d2, 1, t, 1,
                               spatial_thresh, temporal_thresh,
                               max_components, consec_failures,
                               max_iters_main, max_iters_init, tol,
                               NULL, True, True)
        result = pmd(&Y[0], NULL, &U[0], &V[0], parms)
        del parms
        return result


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
        # return pmd(d1, d2, d_sub, t, t_sub, &Y[0], &Y_ds[0], &U[0], &V[0],
        #            spatial_thresh, temporal_thresh,
        #            max_components, consec_failures,
        #            max_iters_main, max_iters_init, tol);

        parms = new PMD_params(d1, d2, d_sub, t, t_sub,
                               spatial_thresh, temporal_thresh,
                               max_components, consec_failures,
                               max_iters_main, max_iters_init, tol,
                               NULL, True, True)
        result = pmd(&Y[0], &Y_ds[0], &U[0], &V[0], parms)
        del parms
        return result



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
        # batch_pmd(bheight, bwidth, d_sub, t, t_sub, num_blocks,
        #           Rp, Rp_ds, Up, Vp, &K[0],
        #           spatial_thresh, temporal_thresh,
        #           max_components, consec_failures,
        #           max_iters_main, max_iters_init, tol)

        parms = new PMD_params(bheight, bwidth, d_sub, t, t_sub,
                             spatial_thresh, temporal_thresh,
                             max_components, consec_failures,
                             max_iters_main, max_iters_init, tol,
                             NULL, True, True)
        batch_pmd(Rp, Rp_ds, Up, Vp, &K[0], num_blocks, parms)
        del parms
        
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


cpdef tv_norm(image):
    return np.sum(np.abs(image[:,:-1] - image[:,1:])) + np.sum(np.abs(image[:-1,:] - image[1:,:]))


cpdef spatial_test_statistic(component):
    d1, d2 = component.shape
    return (tv_norm(component) *d1*d2)/ (np.sum(np.abs(component)) * (d1*(d2-1) + d2 * (d1-1)))


cpdef temporal_test_statistic(signal):
    return np.sum(np.abs(signal[2:] + signal[:-2] - 2*signal[1:-1])) / np.sum(np.abs(signal))


cpdef pca_patch(R, bheight=None, bwidth=None, num_frames=None, spatial_thresh=None, temporal_thresh=None, 
                max_components=50, consec_failures=3, n_iter=7):
    """ """
    
    # Preallocate Space For Outputs
    U = np.zeros((bheight*bwidth, max_components), dtype=np.float64)
    V = np.zeros((max_components, num_frames), dtype=np.float64)

    if np.sum(np.abs(R)) <= 0:
        return U, V, 0
    # Run SVD on the patch
    Ub, sb, Vtb = svd(R, n_components=max_components, n_iter=n_iter)

    # Iteratively Test & discard components
    fails = 0
    K = 0
    for k in range(max_components):
        spatial_stat = spatial_test_statistic(Ub[:,k].reshape((bheight,bwidth), order='F'))
        temporal_stat = temporal_test_statistic(Vtb[k,:])
        if (spatial_stat > spatial_thresh or
            temporal_stat > temporal_thresh):
            fails += 1
            if fails >= consec_failures:
                break
        else:
            U[:,K] = Ub[:,k]
            V[K,:] = Vtb[k,:] * sb[k]
            fails = 0
            K += 1
    return U, V, K


cpdef pca_decompose(const int d1,
                    const int d2,
                    const int t,
                    double[:, :, ::1] Y,
                    const int bheight,
                    const int bwidth,
                    const double spatial_thresh,
                    const double temporal_thresh,
                    const size_t max_components,
                    const size_t consec_failures):
    """ Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function
     factor_patch with OpenMP directives to parallelize batch processing."""

    # Assert Evenly Divisible FOV/Block Dimensions
    assert d1 % bheight == 0 , FOV_BHEIGHT_WARNING
    assert d2 % bwidth == 0 , FOV_BWIDTH_WARNING

    # Initialize Counters
    cdef size_t iu, ku
    cdef int i, j, k, b, bi, bj
    cdef int nbi = int(d1/bheight)
    cdef int nbj = int(d2/bwidth)
    cdef int num_blocks = nbi * nbj

    # Compute block-start indices and spatial cutoff
    indices = np.transpose([np.tile(range(nbi), nbj), np.repeat(range(nbj), nbi)])

    cdef size_t good, fails
    cdef double spatial_stat
    cdef double temporal_stat

    # Copy Residual For Multiprocessing
    R = []
    for bj in range(nbj):
        for bi in range(nbi):
            R.append(np.reshape(Y[(bi * bheight):((bi+1) * bheight),
                                  (bj * bwidth):((bj+1) * bwidth), :],
                                (bheight*bwidth, t), order='F'))

    # Process In Parallel
    try:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = pool.map(partial(pca_patch, 
                                   bheight=bheight, bwidth=bwidth, num_frames=t,
                                   spatial_thresh=spatial_thresh, 
                                   temporal_thresh=temporal_thresh,
                                   max_components=max_components, 
                                   consec_failures=consec_failures),
                           R)
    finally:
        pool.close()
        pool.join()


    # Format Components & Return To Numpy Array
    U, V, K = zip(*results) 
    return (np.array(U).reshape((num_blocks, bheight, bwidth, max_components), order='F'),
            np.array(V), np.array(K).astype(np.uint64), indices.astype(np.uint64))

    
cpdef overlapping_pca_decompose(const int d1, 
                                  const int d2, 
                                  const int t,
                                  double[:, :, ::1] Y, 
                                  const int bheight,
                                  const int bwidth,
                                  const double spatial_thresh,
                                  const double temporal_thresh,
                                  const size_t max_components,
                                  const size_t consec_failures):
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
    I['no_skew']['full'] = pca_decompose(d1, d2, t, Y, bheight, bwidth,
                                           spatial_thresh,temporal_thresh,
                                           max_components, consec_failures)
 
    # ---------- Vertical Skew -----------
    # Full Blocks
    U['vert_skew']['full'],\
    V['vert_skew']['full'],\
    K['vert_skew']['full'],\
    I['vert_skew']['full'] = pca_decompose(d1 - bheight, d2, t, 
                                             Y[hbheight:d1-hbheight,:,:], 
                                             bheight, bwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # wide half blocks
    U['vert_skew']['half'],\
    V['vert_skew']['half'],\
    K['vert_skew']['half'],\
    I['vert_skew']['half'] = pca_decompose(bheight, d2, t, 
                                             np.vstack([Y[:hbheight,:,:], 
                                                        Y[d1-hbheight:,:,:]]),
                                             hbheight, bwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)
    
    # --------------Horizontal Skew---------- 
    # Full Blocks
    U['horz_skew']['full'],\
    V['horz_skew']['full'],\
    K['horz_skew']['full'],\
    I['horz_skew']['full'] = pca_decompose(d1, d2 - bwidth, t, 
                                             Y[:, hbwidth:d2-hbwidth,:], 
                                             bheight, bwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # tall half blocks
    U['horz_skew']['half'],\
    V['horz_skew']['half'],\
    K['horz_skew']['half'],\
    I['horz_skew']['half'] = pca_decompose(d1, bwidth, t, 
                                             np.hstack([Y[:,:hbwidth,:],
                                                        Y[:,d2-hbwidth:,:]]),
                                             bheight, hbwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # -------------Diagonal Skew---------- 
    # Full Blocks
    U['diag_skew']['full'],\
    V['diag_skew']['full'],\
    K['diag_skew']['full'],\
    I['diag_skew']['full'] = pca_decompose(d1 - bheight, d2 - bwidth, t, 
                                             Y[hbheight:d1-hbheight,
                                               hbwidth:d2-hbwidth, :], 
                                             bheight, bwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # tall half blocks
    U['diag_skew']['thalf'],\
    V['diag_skew']['thalf'],\
    K['diag_skew']['thalf'],\
    I['diag_skew']['thalf'] = pca_decompose(d1 - bheight, bwidth, t, 
                                             np.hstack([Y[hbheight:d1-hbheight,
                                                          :hbwidth, :],
                                                        Y[hbheight:d1-hbheight,
                                                          d2-hbwidth:, :]]),
                                             bheight, hbwidth, 
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures)

    # wide half blocks
    U['diag_skew']['whalf'],\
    V['diag_skew']['whalf'],\
    K['diag_skew']['whalf'],\
    I['diag_skew']['whalf'] = pca_decompose(bheight, d2 - bwidth, t, 
                                             np.vstack([Y[:hbheight, 
                                                          hbwidth:d2-hbwidth,
                                                          :], 
                                                        Y[d1-hbheight:,
                                                          hbwidth:d2-hbwidth,
                                                          :]]),
                                             hbheight, bwidth,
                                             spatial_thresh, temporal_thresh,
                                             max_components, consec_failures) 

    # Corners
    U['diag_skew']['quarter'],\
    V['diag_skew']['quarter'],\
    K['diag_skew']['quarter'],\
    I['diag_skew']['quarter'] = pca_decompose(bheight, bwidth, t, 
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
                                                max_components, consec_failures)

    # Return Weighting Matrix For Reconstruction
    return U, V, K, I, W


# Temporary TODO: Optimize, streamline, & add more options (multiple simulations when block
#  size large relative to FOV)

def determine_thresholds(mov_dims,
                         block_dims,
                         num_components,
                         max_iters_main, max_iters_init, tol, 
                         d_sub, t_sub,
                         conf, plot):
    
    # Simulate Noise Movie
    noise_mov = np.ascontiguousarray(np.reshape(np.random.randn(np.prod(mov_dims)), mov_dims))
    
    # Perform Blockwise PMD Of Noise Matrix In Parallel
    spatial_components,\
    temporal_components,\
    block_ranks,\
    block_indices = batch_decompose(mov_dims[0], mov_dims[1], mov_dims[2],
                                    noise_mov, block_dims[0], block_dims[1],
                                    1e3, 1e3,
                                    num_components, num_components,
                                    max_iters_main, max_iters_init, tol,
                                    d_sub=d_sub, t_sub=t_sub)
    
    # Gather Test Statistics
    spatial_stat = []
    temporal_stat = []
    num_blocks = int((mov_dims[0] / block_dims[0]) * (mov_dims[1] / block_dims[1]))
    for block_idx in range(num_blocks): 
        for k in range(int(block_ranks[block_idx])):
            spatial_stat.append(spatial_test_statistic(spatial_components[block_idx,:,:,k]))
            temporal_stat.append(temporal_test_statistic(temporal_components[block_idx,k,:]))

    # Compute Thresholds
    spatial_thresh =  np.percentile(spatial_stat, conf)
    temporal_thresh = np.percentile(temporal_stat, conf)
    
    if plot:
        fig, ax = plt.subplots(2,2,figsize=(8,8))
        ax[0,0].scatter(spatial_stat, temporal_stat, marker='x', c='r', alpha = .2)
        ax[0,0].axvline(spatial_thresh)
        ax[0,0].axhline(temporal_thresh)
        ax[0,1].hist(temporal_stat, bins=20, color='r')
        ax[0,1].axvline(temporal_thresh)
        ax[0,1].set_title("Temporal Threshold: {}".format(temporal_thresh))
        ax[1,0].hist(spatial_stat, bins=20, color='r')
        ax[1,0].axvline(spatial_thresh)
        ax[1,0].set_title("Spatial Threshold: {}".format(spatial_thresh))
        plt.show()
    
    return spatial_thresh, temporal_thresh


# -----------------------------------------------------------------------------#
# --------------------------- Func Wrappers -----------------------------------#
# -----------------------------------------------------------------------------#

# Funct wrapper of batch_decompose & overlapping_batch_decompose, taking an extra
# input value "overlap" (0/1) to indicate whether applying overlapping or not,
# return the result in a tuple
cpdef pmd_batch_decompose(const int d1,
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
                          int t_sub = 1,
                          int overlap = 0):
    if not overlap:
        return (batch_decompose(d1,
                      d2,
                      t,
                      Y,
                      bheight,
                      bwidth,
                      spatial_thresh,
                      temporal_thresh,
                      max_components,
                      consec_failures,
                      max_iters_main,
                      max_iters_init,
                      tol,
                      d_sub,
                      t_sub))
    else:
        return (overlapping_batch_decompose(d1,
                      d2,
                      t,
                      Y,
                      bheight,
                      bwidth,
                      spatial_thresh,
                      temporal_thresh,
                      max_components,
                      consec_failures,
                      max_iters_main,
                      max_iters_init,
                      tol,
                      d_sub,
                      t_sub))

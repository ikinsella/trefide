# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import os

import numpy as np
cimport numpy as np

from libc.stdlib cimport abort, calloc, malloc, free
from cython.parallel import parallel, prange
from trefide.pmd import weighted_recompose


# -----------------------------------------------------------------------------#
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":

    void downsample_1d(const int t, 
                       const int ds, 
                       const double* v, 
                       double* v_ds) nogil


    void downsample_2d(const int d1, 
                       const int d2, 
                       const int ds, 
                       const double* u, 
                       double* u_ds) nogil

    void downsample_3d(const int d1, 
                       const int d2, 
                       const int d_sub, 
                       const int t, 
                       const int t_sub, 
                       double *Y, 
                       double *Y_ds) nogil

    void upsample_1d_inplace(const int t, 
                             const int ds, 
                             double* v) nogil


    void upsample_1d(const int t, 
                     const int ds, 
                     double* v, 
                     const double* v_ds) nogil

    void upsample_2d(const int d1, 
                     const int d2, 
                     const int ds, 
                     double* u, 
                     double* u_ds) nogil

    size_t decimated_pmd(const int d1, 
                         const int d1_ds,
                         const int d2, 
                         const int d2_ds,
                         const int t,
                         const int t_ds,
                         double* R, 
                         double* R_ds,
                         double* U,
                         double* V,
                         const double lambda_tv,
                         const size_t max_components,
                         const size_t consec_failures,
                         const size_t max_iters,
                         const size_t max_iters_ds,
                         const double tol) nogil

    void decimated_batch_pmd(const int bheight,
                             const int bheight_ds, 
                             const int bwidth, 
                             const int bwidth_ds, 
                             const int t,
                             const int t_ds,
                             const int b,
                             double** R, 
                             double** R_ds, 
                             double** U,
                             double** V,
                             size_t* K,
                             const double lambda_tv,
                             const size_t max_components,
                             const size_t consec_failures,
                             const size_t max_iters,
                             const size_t max_iters_ds,
                             const double tol) nogil

# -----------------------------------------------------------------------------#
# ---------------------------- Downsampling Wrappers --------------------------#
# -----------------------------------------------------------------------------#


cpdef double[::1] downsample_signal(const int t, 
                                      const int ds,
                                      double[::1] V):
    """ Downsample Image In Each Dimension By Factor Of ds """

    # Assert Dimensions Match
    assert t % ds == 0, "Signal length must be divisible by downsampling factor."

    # Declare & Intiialize Local Variables
    cdef int t_ds = t / ds

    # Allocate Space For Downsampled Singal
    cdef double[::1] V_ds = np.zeros(t_ds, dtype=np.float64)

    # Call C-Routines From Trefide
    with nogil:
        downsample_1d(t, ds, &V[0], &V_ds[0])

    return np.asarray(V_ds)


cpdef double[::1,:] downsample_image(const int d1, 
                                     const int d2, 
                                     const int ds,
                                     double[::1,:] U):
    """ Downsample Image In Each Dimension By Factor Of ds """

    # Assert Dimensions Match
    assert d1 % ds == 0, "Height of image must be divisible by downsampling factor."
    assert d2 % ds == 0, "Width of image must be divisible by downsampling factor."

    # Declare & Intiialize Local Variables
    cdef int d1_ds = d1 / ds
    cdef int d2_ds = d2 / ds

    # Allocate Space For Downsampled Image
    cdef double[::1] U_ds = np.zeros(d1_ds * d2_ds, dtype=np.float64)

    # Call C-Routines From Trefide
    with nogil:
        downsample_2d(d1, d2, ds, &U[0,0], &U_ds[0])

    return np.reshape(U_ds, (d1_ds, d2_ds), order='F')


cpdef double[::1,:,:] downsample_video(const int d1, 
                                       const int d2, 
                                       const int d_sub, 
                                       const int t, 
                                       const int t_sub, 
                                       double[::1,:,:] Y):
    """ Downsample (d1xd2xt) movie in row-major order """

    # Assert Dimensions Match
    assert d1 % d_sub == 0, "Height of FOV must be divisible by spatial downsampling factor."
    assert d2 % d_sub == 0, "Width of FOV must be divisible by spatial downsampling factor."
    assert t % t_sub == 0, "Num Frames must be divisible by temporal downsampling factor."
    # Declare & Intialize Local Variables
    cdef int d1_ds = d1 / d_sub
    cdef int d2_ds = d2 / d_sub
    cdef int t_ds = t / t_sub

    # Allocate Space For Output 
    cdef double[::1] Y_ds = np.zeros(d1_ds * d2_ds * t_ds, dtype=np.float64)

    # Call C-Routine From Trefide
    with nogil:
        downsample_3d(d1, d2, d_sub, t, t_sub, &Y[0,0,0], &Y_ds[0]) 

    # Format output
    return np.reshape(Y_ds, (d1_ds, d2_ds, t_ds), order='F')


# -----------------------------------------------------------------------------#
# ---------------------------- Upsampling Wrappers --------------------------#
# -----------------------------------------------------------------------------#


cpdef double[::1] upsample_signal(const int t, 
                                  const int ds,
                                  double[::1] V_ds):
    """ Downsample Image In Each Dimension By Factor Of ds """

    # Assert Dimensions Match
    assert t % ds == 0, "Signal length must be divisible by downsampling factor."

    # Allocate Space For Upsampled Singal
    cdef double[::1] V = np.zeros(t, dtype=np.float64)

    # Call C-Routines From Trefide
    with nogil:
        upsample_1d(t, ds, &V[0], &V_ds[0])

    return np.asarray(V)


cpdef double[::1,:] upsample_image(const int d1, 
                                   const int d2, 
                                   const int ds,
                                   double[::1,:] U_ds):
    """ Downsample Image In Each Dimension By Factor Of ds """

    # Assert Dimensions Match
    assert d1 % ds == 0, "Height of image must be divisible by downsampling factor."
    assert d2 % ds == 0, "Width of image must be divisible by downsampling factor."
    
    # Allocate Space For Upsampled Image
    cdef double[::1] U = np.zeros(d1 * d2, dtype=np.float64)

    # Call C-Routines From Trefide
    with nogil:
        upsample_2d(d1, d2, ds, &U[0], &U_ds[0,0])

    return np.reshape(U, (d1, d2), order='F')


# -----------------------------------------------------------------------------#
# -------------------------- Single-Block Wrapper -----------------------------#
# -----------------------------------------------------------------------------#


cpdef size_t decimated_decompose(const int d1, 
                                 const int d1_ds, 
                                 const int d2, 
                                 const int d2_ds, 
                                 const int t,
                                 const int t_ds,
                                 double[::1] Y, 
                                 double[::1] Y_ds, 
                                 double[::1] U,
                                 double[::1] V,
                                 const double lambda_tv,
                                 const size_t max_components,
                                 const size_t consec_failures,
                                 const size_t max_iters,
                                 const size_t max_iters_ds,
                                 const double tol) nogil:
    """ Wrap the single patch cpp PMD functions """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        return decimated_pmd(d1, d1_ds, d2, d2_ds, t, t_ds, 
                             &Y[0], &Y_ds[0], &U[0], &V[0], 
                             lambda_tv, max_components, 
                             consec_failures, max_iters, max_iters_ds, tol);


# -----------------------------------------------------------------------------#
# --------------------------- Multi-Block Wrappers ----------------------------#
# -----------------------------------------------------------------------------#


cpdef decimated_batch_decompose(const int d1, 
                                const int d2, 
                                const int d_sub,
                                const int t,
                                const int t_sub,
                                double[:, :, ::1] Y, 
                                double[:, :, ::1] Y_ds, 
                                const int bheight,
                                const int bwidth,
                                const double lambda_tv,
                                const size_t max_components,
                                const size_t consec_failures,
                                const size_t max_iters,
                                const size_t max_iters_ds,
                                const double tol):
    """ Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function 
     factor_patch with OpenMP directives to parallelize batch processing."""

    # Assert Evenly Divisible FOV/Block Dimensions
    assert d1 % bheight == 0 , "Input FOV height must be an evenly divisible by block height."
    assert d2 % bwidth == 0 , "Input FOV width must be evenly divisible by block width." 
    assert bheight % d_sub == 0 , "Block height must be evenly divisible by spatial downsampling factor."
    assert bwidth % d_sub == 0 , "Block width must be evenly divisible by spatial downsampling factor."
    assert t % t_sub == 0 , "Num Frames must be evenly divisible by temporal downsampling factor."

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
            Rp_ds[b] = <double *> malloc(bheight / d_sub * bwidth / d_sub * t / t_sub * sizeof(double))
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
 
        for bj in range(nbj):
            for bi in range(nbi):
                for k in range(t_ds):
                    for j in range(bwidth_ds):
                        for i in range(bheight_ds):
                            Rp_ds[bi + (bj * nbi)][i + (j * bheight_ds) + (k * bheight_ds * bwidth_ds)] =\
                                    Y_ds[(bi * bheight_ds) + i, (bj * bwidth_ds) + j, k]

        # Decimate Raw Blocks
        #for b in prange(num_blocks, schedule='guided'):
        #    downsample_3d(bheight, bwidth, d_sub, t, t_sub, Rp[b], Rp_ds[b]) 

        # Factor Blocks In Parallel
        decimated_batch_pmd(bheight, bheight_ds, bwidth, bwidth_ds, t, t_ds, num_blocks, 
                            Rp,  Rp_ds, Up, Vp, &K[0], 
                            lambda_tv, max_components, consec_failures, 
                            max_iters, max_iters_ds, tol)

        # Free Allocated Memory
        for b in range(num_blocks):
            free(Rp[b])
            free(Rp_ds[b])
        free(Rp)
        free(Rp_ds)
        free(Up)
        free(Vp)
            
    # Format Components & Return To Numpy Array
    return (np.asarray(U).reshape((num_blocks, bheight, bwidth, max_components), order='F'), 
            np.asarray(V).reshape((num_blocks, max_components, t), order='C'), 
            np.asarray(K), indices.astype(np.uint64))


# -----------------------------------------------------------------------------#
# --------------------------- Overlapping Wrappers ----------------------------#
# -----------------------------------------------------------------------------#


cpdef overlapping_dbd(const int d1, 
                      const int d2, 
                      const int d_sub,
                      const int t,
                      const int t_sub,
                      double[:, :, ::1] Y, 
                      double[:, :, ::1] Y_ds, 
                      const int bheight,
                      const int bwidth,
                      const double lambda_tv,
                      const size_t max_components,
                      const size_t consec_failures,
                      const size_t max_iters,
                      const size_t max_iters_ds,
                      const double tol):
    """ 4x batch denoiser """

    # Assert Even Blockdims
    assert bheight % 2 == 0 , "Block height must be an even integer."
    assert bwidth % 2 == 0 , "Block width must be an even integer."
    
    # Assert Blockdims Compatible With FOV
    assert d1 % bheight == 0 , "Input FOV height must be an evenly divisible by block height."
    assert d2 % bwidth == 0 , "Input FOV width must be evenly divisible by block width."

    # Assert Blocksizes Compatible with Downsampling
    assert bheight % d_sub == 0 , "Block height must be evenly divisible by spatial downsampling factor."
    assert bwidth % d_sub == 0 , "Block width must be evenly divisible by spatial downsampling factor."
    assert bheight / 2 % d_sub == 0 , "Half block height must be evenly divisible by spatial downsampling factor for overlapping decomposition."
    assert bwidth / 2 % d_sub == 0 , "Half block width must be evenly divisible by spatial downsampling factor for overlapping decomposition."
    assert t % t_sub == 0 , "Num Frames must be evenly divisible by temporal downsampling factor."
    
    # Declare internal vars
    cdef int i,j
    cdef int d1_ds = d1 / d_sub
    cdef int d2_ds = d2 / d_sub
    cdef int hbheight = bheight/2
    cdef int hbheight_ds = hbheight / d_sub
    cdef int hbwidth = bwidth/2
    cdef int hbwidth_ds = hbwidth / d_sub
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
    I['no_skew']['full'] = decimated_batch_decompose(d1, d2, d_sub, t, t_sub, Y, Y_ds,
                                                     bheight, bwidth, lambda_tv,
                                                     max_components, consec_failures,
                                                     max_iters, max_iters_ds, tol)

    # ---------- Vertical Skew -----------
    # Full Blocks
    U['vert_skew']['full'],\
    V['vert_skew']['full'],\
    K['vert_skew']['full'],\
    I['vert_skew']['full'] = decimated_batch_decompose(d1 - bheight, d2, d_sub, 
                                                       t, t_sub, 
                                                       Y[hbheight:d1-hbheight,:,:], 
                                                       Y_ds[hbheight_ds:d1_ds-hbheight_ds,:,:],
                                                       bheight, bwidth, 
                                                       lambda_tv,
                                                       max_components, 
                                                       consec_failures,
                                                       max_iters, max_iters_ds,
                                                       tol)

    # wide half blocks
    U['vert_skew']['half'],\
    V['vert_skew']['half'],\
    K['vert_skew']['half'],\
    I['vert_skew']['half'] = decimated_batch_decompose(bheight, d2, d_sub,
                                                       t, t_sub, 
                                                       np.vstack([Y[:hbheight,:,:], 
                                                                  Y[d1-hbheight:,:,:]]),
                                                       np.vstack([Y_ds[:hbheight_ds,:,:], 
                                                                  Y_ds[d1_ds-hbheight_ds:,:,:]]),
                                                       hbheight, bwidth, 
                                                       lambda_tv,
                                                       max_components, 
                                                       consec_failures, 
                                                       max_iters, max_iters_ds, 
                                                       tol)
    
    # --------------Horizontal Skew---------- 
    # Full Blocks
    U['horz_skew']['full'],\
    V['horz_skew']['full'],\
    K['horz_skew']['full'],\
    I['horz_skew']['full'] = decimated_batch_decompose(d1, d2 - bwidth, d_sub,
                                                       t, t_sub, 
                                                       Y[:, hbwidth:d2-hbwidth,:], 
                                                       Y_ds[:, hbwidth_ds:d2_ds-hbwidth_ds,:], 
                                                       bheight, bwidth,
                                                       lambda_tv,
                                                       max_components, 
                                                       consec_failures, 
                                                       max_iters, max_iters_ds,
                                                       tol)

    # tall half blocks
    U['horz_skew']['half'],\
    V['horz_skew']['half'],\
    K['horz_skew']['half'],\
    I['horz_skew']['half'] = decimated_batch_decompose(d1, bwidth, d_sub,
                                                       t, t_sub, 
                                                       np.hstack([Y[:,:hbwidth,:],
                                                                  Y[:,d2-hbwidth:,:]]),
                                                       np.hstack([Y_ds[:,:hbwidth_ds,:],
                                                                  Y_ds[:,d2_ds-hbwidth_ds:,:]]),
                                                       bheight, hbwidth, 
                                                       lambda_tv, 
                                                       max_components, 
                                                       consec_failures, 
                                                       max_iters, max_iters_ds,
                                                       tol)

    # -------------Diagonal Skew---------- 
    # Full Blocks
    U['diag_skew']['full'],\
    V['diag_skew']['full'],\
    K['diag_skew']['full'],\
    I['diag_skew']['full'] = decimated_batch_decompose(d1 - bheight, d2 - bwidth, d_sub,
                                                       t, t_sub, 
                                                       Y[hbheight:d1-hbheight,
                                                         hbwidth:d2-hbwidth, :], 
                                                       Y_ds[hbheight_ds:d1_ds-hbheight_ds,
                                                            hbwidth_ds:d2_ds-hbwidth_ds, :], 
                                                       bheight, bwidth,
                                                       lambda_tv,
                                                       max_components, 
                                                       consec_failures,
                                                       max_iters, max_iters_ds,
                                                       tol)

    # tall half blocks
    U['diag_skew']['thalf'],\
    V['diag_skew']['thalf'],\
    K['diag_skew']['thalf'],\
    I['diag_skew']['thalf'] = decimated_batch_decompose(d1 - bheight, bwidth, d_sub, 
                                                        t, t_sub, 
                                                        np.hstack([Y[hbheight:d1-hbheight,
                                                                     :hbwidth, :],
                                                                   Y[hbheight:d1-hbheight,
                                                                     d2-hbwidth:, :]]),
                                                        np.hstack([Y_ds[hbheight_ds:d1_ds-hbheight_ds,
                                                                        :hbwidth_ds, :],
                                                                   Y_ds[hbheight_ds:d1_ds-hbheight_ds,
                                                                        d2_ds-hbwidth_ds:, :]]),
                                                        bheight, hbwidth,
                                                        lambda_tv, 
                                                        max_components, 
                                                        consec_failures, 
                                                        max_iters, max_iters_ds,
                                                        tol)

    # wide half blocks
    U['diag_skew']['whalf'],\
    V['diag_skew']['whalf'],\
    K['diag_skew']['whalf'],\
    I['diag_skew']['whalf'] = decimated_batch_decompose(bheight, d2 - bwidth, d_sub, 
                                                        t, t_sub, 
                                                        np.vstack([Y[:hbheight, 
                                                                     hbwidth:d2-hbwidth,
                                                                     :], 
                                                                   Y[d1-hbheight:,
                                                                     hbwidth:d2-hbwidth,
                                                                     :]]),
                                                        np.vstack([Y_ds[:hbheight_ds, 
                                                                        hbwidth_ds:d2_ds-hbwidth_ds,
                                                                        :], 
                                                                   Y_ds[d1_ds-hbheight_ds:,
                                                                        hbwidth_ds:d2_ds-hbwidth_ds,
                                                                        :]]),
                                                         hbheight, bwidth,
                                                         lambda_tv,
                                                         max_components, 
                                                         consec_failures, 
                                                         max_iters, 
                                                         max_iters_ds, 
                                                         tol) 

    # Corners
    U['diag_skew']['quarter'],\
    V['diag_skew']['quarter'],\
    K['diag_skew']['quarter'],\
    I['diag_skew']['quarter'] = decimated_batch_decompose(bheight, bwidth, d_sub,  
                                                          t, t_sub, 
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
                                                          np.hstack([
                                                              np.vstack([Y_ds[:hbheight_ds,
                                                                              :hbwidth_ds,
                                                                              :], 
                                                                         Y_ds[d1_ds-hbheight_ds:,
                                                                              :hbwidth_ds,
                                                                              :]]),
                                                              np.vstack([Y_ds[:hbheight_ds,
                                                                              d2_ds-hbwidth_ds:,
                                                                              :], 
                                                                         Y_ds[d1_ds-hbheight_ds:,
                                                                              d2_ds-hbwidth_ds:,
                                                                              :]])
                                                                    ]),
                                                          hbheight, hbwidth, 
                                                          lambda_tv, 
                                                          max_components, 
                                                          consec_failures, 
                                                          max_iters, max_iters_ds, 
                                                          tol)

    # Return Weighting Matrix For Reconstruction
    return U, V, K, I, W

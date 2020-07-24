# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: language_level=3
import os
import sys
import multiprocessing
from functools import partial

import numpy as np
cimport numpy as np
import scipy

from libc.stdlib cimport malloc, free
from sklearn.utils.extmath import randomized_svd as svd
import matplotlib.pyplot as plt
from libcpp cimport bool

FOV_block_height_WARNING = "Input FOV height must be an evenly divisible by block height."
FOV_block_width_WARNING = "Input FOV width must be evenly divisible by block width."
DSUB_block_height_WARNING = "Block height must be evenly divisible by spatial downsampling factor."
DSUB_block_width_WARNING = "Block width must be evenly divisible by spatial downsampling factor."
TSUB_FRAMES_WARNING = "Num Frames must be evenly divisible by temporal downsampling factor."
OVERLAP_WARNING = "Spatial block dimensions must be even for an overlapping decomposition."

# -----------------------------------------------------------------------------#
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":

    cdef cppclass PMD_params:
        PMD_params(
            const int _block_height,
            const int _block_width,
            int _spatial_ds_factor,
            const int _t,
            int _temporal_ds_factor,
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

    size_t pmd(
            double* R,
            double* R_ds,
            double* U,
            double* V,
            PMD_params *pars) nogil

    void batch_pmd(double** Up,
                   double** Vp,
                   size_t* K,
                   const int num_blocks,
                   PMD_params* pars,
                   size_t* indices) nogil

    void downsample_3d(const int fov_height,
                       const int fov_width,
                       const int spatial_ds_factor,
                       const int t,
                       const int temporal_ds_factor,
                       const double *Y,
                       double *Y_ds) nogil


# -----------------------------------------------------------------------------#
# -------------------------- Single-Block Wrapper -----------------------------#
# -----------------------------------------------------------------------------#


cpdef size_t decompose(const int fov_height, const int fov_width, const int t, double[::1] Y,
        double[::1] U, double[::1] V, const double spatial_thresh, const double
        temporal_thresh, const size_t max_components, const size_t
        consec_failures, const size_t max_iters_main, const size_t
        max_iters_init, const double tol, bool enable_temporal_denoiser = True,
        bool enable_spatial_denoiser = True) nogil:
    """Apply TF/TV Penalized Matrix Decomposition (PMD) to factor a
       column major formatted video into spatial and temporal components.

    Parameters
    ----------
    fov_height :
        height of video
    fov_width :
        width of video
    t :
        frames of video
    Y :
        video data of shape (fov_height x fov_width) x t
    U :
        decomposed spatial component matrix
    V :
        decomposed temporal component matrix
    spatial_thresh :
        spatial threshold
    temporal_thresh :
        temporal threshold
    max_components :
        maximum number of components
    consec_failures :
        number of failures before stopping
    max_iters_main :
        maximum number of iterations refining a component
    max_iters_init :
        maximum number of iterations refining a component during decimated
        initialization
    tol : convergence tolerence
    enable_temporal_denoiser :
        whether enable temporal denoiser, True by default
    enable_spatial_denoiser :
        whether enable spatial denoiser, True by default

    Returns
    -------
    result :
        rank of the compressed video/patch
    """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        parms = new PMD_params(fov_height, fov_width, 1, t, 1,
                               spatial_thresh, temporal_thresh,
                               max_components, consec_failures,
                               max_iters_main, max_iters_init, tol,
                               NULL,
                               enable_temporal_denoiser,
                               enable_spatial_denoiser)
        result = pmd(&Y[0], NULL, &U[0], &V[0], parms)
        del parms
        return result


cpdef size_t decimated_decompose(const int fov_height,
                                 const int fov_width,
                                 int spatial_ds_factor,
                                 const int t,
                                 int temporal_ds_factor,
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
                                 const double tol,
                                 bool enable_temporal_denoiser = True,
                                 bool enable_spatial_denoiser = True) nogil:
    """ Apply decimated TF/TV Penalized Matrix Decomposition (PMD) to factor a
    column major formatted video into spatial and temporal components.

    Parameters
    ----------
    fov_height :
        height of video
    fov_width :
        width of video
    spatial_ds_factor :
        spatial downsampling factor
    t :
        frames of video
    temporal_ds_factor :
        temporal downsampling factor
    Y :
        video data of shape (fov_height x fov_width) x t
    Y_ds :
        downsampled video data
    U :
        decomposed spatial matrix
    V :
        decomposed temporal matrix
    spatial_thresh :
        spatial threshold,
    temporal_thresh :
        temporal threshold,
    max_components :
        maximum number of components,
    consec_failures :
        number of failures before stopping
    max_iters_main :
        maximum number of iterations refining a component
    max_iters_init :
        maximum number of iterations refining a component during decimated initialization
    tol :
        convergence tolerence
    enable_temporal_denoiser :
        whether enable temporal denoiser, True by default
    enable_spatial_denoiser :
        whether enable spatial denoiser, True by default

    Returns
    -------
    result :
        rank of the compressed video/patch
    """

    # Turn Off Gil To Take Advantage Of Multithreaded MKL Libs
    with nogil:
        parms = new PMD_params(fov_height, fov_width, spatial_ds_factor, t, temporal_ds_factor,
                               spatial_thresh, temporal_thresh,
                               max_components, consec_failures,
                               max_iters_main, max_iters_init, tol,
                               NULL,
                               enable_temporal_denoiser,
                               enable_spatial_denoiser)
        result = pmd(&Y[0], &Y_ds[0], &U[0], &V[0], parms)
        del parms
        return result


# -----------------------------------------------------------------------------#
# --------------------------- Multi-Block Wrappers ----------------------------#
# -----------------------------------------------------------------------------#

def batch_decompose(movie_shape,
                    block_shape,
                    const double spatial_thresh,
                    const double temporal_thresh,
                    const size_t max_components,
                    const size_t consec_failures,
                    const size_t max_iters_main,
                    const size_t max_iters_init,
                    const double tol,
                    int spatial_ds_factor=1,
                    int temporal_ds_factor=1,
                    bool enable_temporal_denoiser=True,
                    bool enable_spatial_denoiser=True,
                    overlap=True):
    """ Apply TF/TV Penalized Matrix Decomposition (PMD) in batch to factor a
    column major formatted video into spatial and temporal components.

    Wrapper for the .cpp parallel_factor_patch which wraps the .cpp function
    factor_patch with OpenMP directives to parallelize batch processing.

    Parameters
    ----------
    movie_shape : tuple (fov_height, fov_width, num_of_frames)

    block_shape : tuple (block_height, block_width, block_frames)

    spatial_thresh : float
        spatial threshold,
    temporal_thresh : float
        temporal threshold,
    max_components :
        maximum number of components,
    consec_failures :
        number of failures before stopping
    max_iters_main :
        maximum number of iterations refining a component
    max_iters_init :
        maximum number of iterations refining a component during decimated
        initialization
    tol :
        convergence tolerence
    spatial_ds_factor :
        spatial downsampling factor
    temporal_ds_factor :
        temporal downsampling factor
    enable_temporal_denoiser :
        whether enable temporal denoiser, True by default
    enable_spatial_denoiser :
        whether enable spatial denoiser, True by default
    overlap : bool, default: True
        Whether to use spatially overlapping blocks

    Returns
    -------
    compact_spatial :
        Compact representation of nonzero elements of the spatial components,
        i.e., U
    temporal :
        Temporal components matrix, i.e., V
    block_ranks :
        Rank of each spatial block
    block_coords :
        Actual pixel index for each corresponding patch grid
    """
    cdef int fov_height = movie_shape[0]
    cdef int fov_width = movie_shape[1]
    cdef int num_frames = movie_shape[2]

    cdef int block_height = block_shape[0]
    cdef int block_width = block_shape[1]

    # Assert Evenly Divisible FOV/Block Dimensions
    if fov_height % block_height != 0:
        raise ValueError(FOV_block_height_WARNING+" fov_height: {} block_height: {}".format(fov_height, block_height))
    if fov_width % block_width != 0:
        raise ValueError(FOV_block_width_WARNING+" fov_width: {} spatial_ds_factor: {}".format(fov_width, block_width))
    if block_height % spatial_ds_factor != 0:
        raise ValueError(DSUB_block_height_WARNING+" block_height {}: spatial_ds_factor: {}".format(block_height, spatial_ds_factor))
    if block_width % spatial_ds_factor != 0:
        raise ValueError(DSUB_block_width_WARNING+" block_width {}: spatial_ds_factor: {}".format(block_width, spatial_ds_factor))
    if num_frames % temporal_ds_factor != 0:
        raise ValueError(TSUB_FRAMES_WARNING+" t {}: temporal_ds_factor: {}".format(num_frames, temporal_ds_factor))
    if overlap and (block_height % 2 != 0 or block_width % 2 != 0):
        raise ValueError("{} overlap: {} block_shape: {}".format(OVERLAP_WARNING, overlap, block_shape))

    # Compute block-start indices and spatial cutoff
    cdef int nbi = fov_height // block_height
    cdef int nbj = fov_width // block_width
    cdef int num_blocks = nbi * nbj

    incr_height = block_height
    incr_width = block_width

    if overlap:
        incr_height //= 2
        incr_width //= 2

    cdef size_t[:, ::1] indices = np.array(
        [coord for coord in zip(np.tile(range(0, fov_height, incr_height), nbj),
                                np.repeat(range(0, fov_width, incr_width), nbi))],
        dtype=np.uint64
    )

    # Preallocate Space For Outputs
    cdef double[:,::1] compact_spatial = np.zeros((num_blocks, block_height * block_width * max_components), dtype=np.float64)
    cdef double[:,::1] temporal = np.zeros((num_blocks, num_frames * max_components), dtype=np.float64)
    cdef size_t[::1] block_ranks = np.empty(num_blocks, dtype=np.uint64)

    # Allocate Input Pointers
    cdef double** Vp = <double **> malloc(num_blocks * sizeof(double*))
    cdef double** Up = <double **> malloc(num_blocks * sizeof(double*))

    # Release Gil Prior To Referencing Address & Calling Multithreaded Code
    with nogil:

        # Assign Pre-allocated Output Memory To Pointer Array & Allocate
        # Residual Pointers
        for b in range(num_blocks):
            Up[b] = &compact_spatial[b, 0]
            Vp[b] = &temporal[b, 0]

        params = new PMD_params(block_height, block_width, spatial_ds_factor,
                                num_frames, temporal_ds_factor, spatial_thresh,
                                temporal_thresh, max_components,
                                consec_failures, max_iters_main, max_iters_init,
                                tol, NULL, enable_temporal_denoiser,
                                enable_spatial_denoiser)

        batch_pmd(Up, Vp, &block_ranks[0], num_blocks, params, &indices[0, 0])

        del params

    free(Up)
    free(Vp)

    # Format Components & Return To Numpy Array
    return (np.asarray(compact_spatial).reshape((num_blocks, block_height, block_width, max_components), order='F'),
            np.asarray(temporal).reshape((num_blocks, max_components, num_frames), order='C'),
            np.asarray(block_ranks), np.asarray(indices))


def reformat_spatial(compact_spatial,
                     block_ranks,
                     block_coords,
                     block_weights=None):
    """

    Constructs CSR spatial matrix from compact results of PMD such that
    Y_hat = np.reshape(np.asarray(spatial.dot(temporal)), (fov_height, fov_width, -1))

    Parameters
    ----------
    compact_spatial :
        Compact representation of nonzero elements of the spatial components,
        i.e., U
    block_ranks :
        Rank of each spatial block
    block_coords :
        Actual pixel index for each corresponding patch grid

    Returns
    -------
    sparse_spatial :

    """
    # Precompute Dims & Allocate COO Matrix Accordingly
    bheight, bwidth = compact_spatial.shape[1:3]
    fov_height, fov_width = np.max(block_coords, axis=0) + np.array([bheight, bwidth])
    total_rank = np.sum(block_ranks)
    sparse_spatial = scipy.sparse.lil_matrix((fov_width * fov_height, total_rank), dtype=np.float32)

    # Default Parameter Setting: Constant Weights & Scales, empty cumulative weights
    if block_weights is None:
        block_weights = np.ones((bheight, bwidth))

    fov_weights = np.zeros((fov_height, fov_width), dtype=np.float32)

    # Accumulate Scaled Blocks
    r0 = 0
    for bdx, rank in enumerate(block_ranks):
        # Compute Coord Offsets (wrt. FOV) & Rank of Block
        ydx = block_coords[bdx, 0]
        xdx = block_coords[bdx, 1]

        if rank > 0:
            # Copy Each Block Into Unraveled Full FOV
            row_inc = np.reshape((ydx + np.arange(bheight)) * fov_width, (bheight, 1))
            insert_idx = np.tile(np.arange(bwidth) + xdx, (bheight, 1))
            insert_idx = np.reshape(insert_idx + row_inc, (-1,))
            sparse_spatial[insert_idx, r0:r0 + rank] = np.reshape(
                np.multiply(
                    np.reshape(compact_spatial[bdx, :, :, :rank], (bheight, bwidth, rank)),
                    block_weights[:, :, None]
                ),
                (bheight * bwidth, rank)
            )

        # Increment Cumulative Weight Image & Rank Counter
        fov_weights[ydx:ydx+bheight, xdx:xdx+bwidth] += block_weights
        r0 += rank

    # Using FOV weights to normalize
    normalizing_weights = scipy.sparse.diags([(1 / fov_weights).ravel()], [0])
    sparse_spatial = normalizing_weights.dot(sparse_spatial.tocsr())

    # Return CSR For Efficient Pixelwise Slicing (Used In HemoCorr)
    return sparse_spatial

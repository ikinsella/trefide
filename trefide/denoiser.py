
import numpy as np

from . import pmd


def batch_decompose(d1,
                    d2,
                    nchan,
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
                    d_sub = 1,
                    t_sub = 1,
                    enable_temporal_denoiser = True,
                    enable_spatial_denoiser = True):
    """ Apply TF/TV Penalized Matrix Decomposition (PMD) in batch to factor a
        column major formatted video into spatial and temporal components.

        Wrapper of pmd.batch_decompose() function.

        Parameters:
            d1: height of video 
            d2: width of video
            t: frames of video
            Y: video data of shape (d1 x d2) x t
            bheight: height of video block
            bwidth: width of video block
            spatial_thresh: spatial threshold,
            temporal_thresh: temporal threshold,
            max_components: maximum number of components,
            consec_failures: number of failures before stopping
            max_iters_main: maximum number of iterations refining a component
            max_iters_init: maximum number of iterations refining a component during decimated initializatio
            tol: convergence tolerence
            d_sub: spatial downsampling factor
            t_sub: temporal downsampling factor
            enable_temporal_denoiser: whether enable temporal denoiser, True by default
            enable_spatial_denoiser: whether enable spatial denoiser, True by default

        Return:
            U: spatial components matrix
            V: temporal components matrix
            K: rank of each patch
            indices: location/index inside of patch grid
    """

    return pmd.batch_decompose(d1,
                               d2,
                               nchan,
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
                               t_sub,
                               enable_temporal_denoiser,
                               enable_spatial_denoiser)


def batch_recompose(denoiser_outputs, weights=None):

    """ Reconstruct A Denoised Movie from components returned by batch_decompose.

    Parameter:
        denoiser_outputs: a tuple of (U: spatial component matrix
                                      V: temporal component matrix
                                      K: rank of each patch
                                      indices: location/index inside of patch grid),
                        output from batch_decompose
        weights: weights to combine components from different overlapping decomposition

    Return:
        Y_den: denoised video data
    """

    #Preallocate
    bheight, bwidth, nchan = denoiser_outputs[0].shape[1:4]
    T = denoiser_outputs[1].shape[-1]
    nbh, nbw = np.max(denoiser_outputs[-1], axis=0)
    Y_den = np.zeros((int(nbh+1)*bheight, int(nbw+1)*bwidth, nchan, T))
    if weights is None:
        weights = np.ones((bheight,bwidth))

    # Accumulate Scaled Blocks
    for bdx, (rank, block_inds) in enumerate(zip(denoiser_outputs[2],
                                                 denoiser_outputs[3])):
        rank = int(rank)
        ydx = int(bheight * block_inds[0])
        xdx = int(bwidth * block_inds[1])
        Y_den[ydx:ydx+bheight,xdx:xdx+bwidth,:,:] = np.dot(
            denoiser_outputs[0][bdx,:,:,:,:rank] * weights[:,:,None,None],
            denoiser_outputs[1][bdx,:rank,:])
    return Y_den


def overlapping_batch_decompose(d1,
                                d2,
                                nchan,
                                T,
                                Y,
                                bheight,
                                bwidth,
                                spatial_thresh,
                                temporal_thresh,
                                max_comp,
                                consec_failures,
                                max_iters_main,
                                max_iters_init,
                                tol,
                                d_sub = 1,
                                t_sub = 1,
                                enable_temporal_denoiser = True,
                                enable_spatial_denoiser = True):

    """ 4x batch denoiser. Apply TF/TV Penalized Matrix Decomposition (PMD) in
    batch to factor a column major formatted video into spatial and temporal
    components.

    Wrapper of batch_decompose function to parallelize batch processing.

    Parameter:
        d1: height of video
        d2: width of video
        t: frames of video
        Y: video data of shape (d1 x d2) x t
        bheight: height of video block
        bwidth: width of video block
        spatial_thresh: spatial threshold,
        temporal_thresh: temporal threshold,
        max_components: maximum number of components,
        consec_failures: number of failures before stopping
        max_iters_main: maximum number of iterations refining a component
        max_iters_init: maximum number of iterations refining a component during decimated initializatio
        tol: convergence tolerence
        d_sub: spatial downsampling factor
        t_sub: temporal downsampling factor
        enable_temporal_denoiser: whether enable temporal denoiser, True by default
        enable_spatial_denoiser: whether enable spatial denoiser, True by default

    Return:
        outs: a list of tuples, with each tuples contains,
            U: spatial components matrix
            V: temporal components matrix
            K: rank of each patch
            I: location/index inside of patch grid
            W: weighting components matrix

    """

    outs = []
    hbheight = int(bheight/2)
    hbwidth = int(bwidth/2)

    #Run Once On OG
    outs.append(batch_decompose(d1, d2, nchan, T,
                                Y, bheight, bwidth,
                                spatial_thresh, temporal_thresh,
                                max_comp,
                                consec_failures,
                                max_iters_main, max_iters_init,
                                tol,
                                d_sub,
                                t_sub,
                                enable_temporal_denoiser,
                                enable_spatial_denoiser
                                ))


    #Run again on vertical offset
    Y_tmp = np.ascontiguousarray(Y[hbheight:-hbheight,:,:,:])
    outs.append(batch_decompose(d1-bheight, d2, nchan, T,
                                Y_tmp, bheight, bwidth,
                                spatial_thresh, temporal_thresh,
                                max_comp,
                                consec_failures,
                                max_iters_main, max_iters_init,
                                tol,
                                d_sub,
                                t_sub,
                                enable_temporal_denoiser,
                                enable_spatial_denoiser
                                ))

    #Run again on horizontal offset
    Y_tmp = np.ascontiguousarray(Y[:,hbwidth:-hbwidth,:,:])
    outs.append(batch_decompose(d1, d2-bwidth, nchan, T,
                                Y_tmp, bheight, bwidth,
                                spatial_thresh, temporal_thresh,
                                max_comp,
                                consec_failures,
                                max_iters_main, max_iters_init,
                                tol,
                                d_sub,
                                t_sub,
                                enable_temporal_denoiser,
                                enable_spatial_denoiser
                                ))

    # Run again on diagonal offset
    Y_tmp = np.ascontiguousarray(Y[hbheight:-hbheight,hbwidth:-hbwidth,:,:])
    outs.append(batch_decompose(d1-bheight, d2-bwidth, nchan, T,
                                Y_tmp, bheight, bwidth,
                                spatial_thresh, temporal_thresh,
                                max_comp,
                                consec_failures,
                                max_iters_main, max_iters_init,
                                tol,
                                d_sub,
                                t_sub,
                                enable_temporal_denoiser,
                                enable_spatial_denoiser
                                ))

    return outs

def overlapping_batch_recompose(outs,
                                d1,
                                d2,
                                bheight,
                                bwidth):

    """ 4x batch denoiser. Reconstruct A Denoised Movie from components
    returned by batch_decompose.

    Parameter:
        outs: decomposed matrix components, output from overlapping_batch_decompose
        d1: height of video
        d2: width of video
        t: time frames of video
        bheight: block height
        bwidth: block width

    Return:
        Y_den: recomposed video data matrix
    """

    hbheight = int(bheight/2)
    hbwidth = int(bwidth/2)

    # Generate Single Quadrant Weights
    ul_weights = np.empty((hbheight, hbwidth), dtype=np.float64)
    for i in range(hbheight):
        for j in range(hbwidth):
            ul_weights[i,j] = min(i, j)+1

    # Construct Full Tile Weights From Quadrant
    tile_weights = np.hstack([np.vstack([ul_weights,
                                         np.flipud(ul_weights)]),
                              np.vstack([np.fliplr(ul_weights),
                                         np.fliplr(np.flipud(ul_weights))])])

    # Construct Full FOV Weights By Repeating
    weights = np.tile(tile_weights, (int(d1/bheight), int(d2/bwidth)))

    # Sum All Weights At Get FOV Pixelwise-Normalization
    cumulative_weights = np.zeros((d1,d2))
    cumulative_weights += weights
    cumulative_weights[hbheight:-hbheight,:] += weights[:-bheight, :]
    cumulative_weights[:,hbwidth:-hbwidth] += weights[:, :-bwidth]
    cumulative_weights[hbheight:-hbheight,hbwidth:-hbwidth] += weights[:-bheight, :-bwidth]

    # Compose Original Tiling
    Y_den = batch_recompose(outs[0], weights=tile_weights)
    # Add Horizontal Offset
    Y_tmp = batch_recompose(outs[1], weights=tile_weights)
    Y_den[hbheight:-hbheight,:,:,:] += Y_tmp
    # Add Vertical Offset
    Y_tmp = batch_recompose(outs[2], weights=tile_weights)
    Y_den[:,hbwidth:-hbwidth,:,:] += Y_tmp
    # Add Diagonal Offset
    Y_tmp = batch_recompose(outs[3], weights=tile_weights)
    Y_den[hbheight:-hbheight,hbwidth:-hbwidth,:,:] += Y_tmp

    # Normalize Movie With recombination weights
    Y_den /= cumulative_weights[:,:,None,None]

    return Y_den

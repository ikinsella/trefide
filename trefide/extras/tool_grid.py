import sys
import numpy as np
#import preprocess_blockSVD as pre_svd
import multiprocessing
import time
import matplotlib.pyplot as plt
#import greedyPCA_SV as gpca
#import greedyPCA as gpca
from math import ceil

from functools import partial
from itertools import product

# compute single mean_th factor for all tiles


def block_split_size(l, n):
    """
    For an array of length l that should be split into n sections,
    calculate the dimension of each section:
    l%n sub-arrays of size l//n +1 and the rest of size l//n
    Input:
    ------
    l:      int
            length of array
    n:      int
            number of section in which an array of size l
            will be partitioned
    Output:
    ------
    d:      np.array (n,)
            length of each partitioned array.
    """
    d = np.zeros((n,)).astype('int')
    cut = l%n
    d[:cut] = l//n+1
    d[cut:] = l//n
    return d


def split_image_into_blocks(image,
                          nblocks=[10,10]):
    """
    Split an image into blocks.

    Parameters:
    ----------
    image:          np.array (d1 x d2 x T)
                    array to be split into nblocks
                    along first two dimensions
    nblocks:        list (2,)
                    parameters to split image across
                    the first two dimensions, respectively

    Outputs
    -------
    blocks:         list,
                    contains nblocks[0]*nblocks[1] number of tiles
                    each of dimensions (d1' x d2' x T)
                    in fortran 'F' order.
    """

    if all(isinstance(n, int) for n in nblocks):
        number_of_blocks = np.prod(nblocks)
    else:
        number_of_blocks = (len(nblocks[0])+1)*(len(nblocks[1])+1)
    blocks = []
    if number_of_blocks != (image.shape[0] * image.shape[1]):
        block_divided_image = np.array_split(image,nblocks[0],axis=0)
        for row in block_divided_image:
            blocks_ = np.array_split(row,nblocks[1],axis=1)
            for block in blocks_:
                blocks.append(np.array(block))
    else:
        blocks = image.flatten()
    return blocks


def vector_offset(array, offset_factor=2):
    """
    Given the dimenions of a matrix (dims), which was
    split row and column wise according to row_array,col_array,
    Calculate the offset in which to split the
    Inputs:
    -------

    Outputs:
    -------
    """
    #x,y = np.meshgrid(row_array[:],col_array[:])
    array_offset = np.ceil(np.divide(np.diff(array),
                                     offset_factor)).astype('int')
    #c_offset = np.ceil(np.divide(np.diff(col_array),2)).astype('int')

    # calculate the dimensions of three off-grid splits
    #row_cut = row_array[:-1]+r_offset
    #col_cut = col_array[:-1]+c_offset

    #dims_rs = dims[0],row_cut[-1]-row_cut[0],dims[2]
    #dims_cs = col_cut[-1]-col_cut[0],dims[1],dims[2]
    #dims_rcs = col_cut[-1]-col_cut[0],row_cut[-1]-row_cut[0],dims[2]
    return array_offset


def tile_grids(dims,
               indiv_grids=True,
               nblocks=[10,10]):
    """
    Input:
    ------

    Output:
    ------
    """

    if all(isinstance(n, int) for n in nblocks):
        d_row = block_split_size(dims[0],nblocks[0])
        d_col = block_split_size(dims[1],nblocks[1])
    else:
        d_row, d_col=nblocks

    if indiv_grids:
        d_row = np.insert(d_row,0,0)
        d_col = np.insert(d_col,0,0)
        return d_row.cumsum(),d_col.cumsum()

    d_row = np.append(d_row,dims[0])
    d_col = np.append(d_col,dims[1])
    d_row = np.diff(np.insert(d_row,0,0))
    d_col = np.diff(np.insert(d_col,0,0))

    number_of_blocks = (len(d_row))*(len(d_col))

    #row_array = np.zeros((number_blocks,))
    #col_array = np.zeros((number_blocks,))
    array = np.zeros((number_of_blocks,2))

    for ii,row in enumerate(product(d_row,d_col)):
        array[ii] = row

    """
    # for each row
    for ii in range(nblocks[0]):
        # split it into cols
        d_col = block_split_size(dims[1],nblocks[1])
        # advance col size
        idx_= ii*nblocks[1]
        # assign the row dim
        row_array[idx_:idx_+nblocks[1]]=d_row[ii]
        # assign the col dim
        col_array[idx_:idx_+nblocks[1]]=d_col
    """

    # return size of row col dimension
    #return np.stack((row_array.astype('int'),col_array.astype('int'))).T
    return array.astype('int')


def offset_tiling_dims(dims,
                      nblocks,
                      offset_case=None):
    """
    """
    row_array, col_array = tile_grids(dims,
                                    nblocks=nblocks)
    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)

    rc0, rc1 = (row_array[1:]-r_offset)[[0,-1]]
    cc0, cc1 = (col_array[1:]-c_offset)[[0,-1]]

    if offset_case is None:
        row_array=row_array[1:-1]
        col_array=col_array[1:-1]

    elif offset_case == 'r':
        dims = rc1-rc0,dims[1],dims[2]
        row_array=row_array[1:-2]
        col_array=col_array[1:-1]

    elif offset_case == 'c':
        dims = dims[0],cc1-cc0,dims[2]
        row_array=row_array[1:-1]
        col_array=col_array[1:-2]

    elif offset_case == 'rc':
        dims = rc1-rc0,cc1-cc0,dims[2]
        row_array=row_array[1:-2]
        col_array=col_array[1:-2]

    else:
        print('Invalid option')

    indiv_dim = tile_grids(dims,
                           nblocks=[row_array,col_array],
                           indiv_grids=False)
    return dims, indiv_dim


def offset_tiling(W,
                  nblocks=[10,10],
                  offset_case=None):
    """
    Given a matrix W, which was split row and column wise
    given row_cut,col_cut, calculate three off-grid splits
    of the same matrix. Each offgrid will be only row-,
    only column-, and row and column-wise.
    Inputs:
    -------
    W:          np.array (d1 x d2 x T)
    r_offset:
    c_offset:
    row_cut:
    col_cut:

    Outputs:
    --------
    W_rs:       list
    W_cs:       list
    W_rcs:      list
    """

    #col_array,row_array = tile_grids(dims,nblocks)

    #r_offset,c_offset = extract_4dx_grid(dims,row_array,col_array)
    dims=W.shape
    row_array,col_array = tile_grids(dims,
                                    nblocks=nblocks)

    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)

    rc0, rc1 = (row_array[1:]-r_offset)[[0,-1]]
    cc0, cc1 = (col_array[1:]-c_offset)[[0,-1]]

    if offset_case is None:
        W_off = split_image_into_blocks(W,
                                        nblocks=nblocks)

    elif offset_case == 'r':
        W = W[rc0:rc1,:,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-2],
                                                 col_array[1:-1]])
    elif offset_case == 'c':
        W = W[:,cc0:cc1,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-1],
                                                 col_array[1:-2]])
    elif offset_case == 'rc':
        W = W[rc0:rc1,cc0:cc1,:]
        W_off = split_image_into_blocks(W,
                                        nblocks=[row_array[1:-2],
                                                 col_array[1:-2]])
    else:
        print('Invalid option')
        W_off = W

    return W_off, W.shape


def denoise_dx_tiles(W,
                    confidence=0.99,
                    dx=1,
                    fudge_factor=1.,
                    greedy=False,
                    maxlag=3,
                    mean_th_factor=1.15,
                    min_rank=1,
                    nblocks=[10,10],
                    snr_threshold=2,
                    U_update=False,
                    verbose=False):
    """
    Given matrix W, denoise it according
    Input:
    ------

    Output:
    ------
    """
    dims = W.shape

    W_ = split_image_into_blocks(W,nblocks=nblocks)

    #########################
    # No offset tiling
    #########################
    if verbose:
        print('Running individual tiles')


    dW_,rank_W_ = run_single(W_,
                            confidence=confidence,
                            fudge_factor=fudge_factor,
                            greedy=greedy,
                            maxlag=maxlag,
                            mean_th_factor=mean_th_factor,
                            min_rank=min_rank,
                            snr_threshold=snr_threshold,
                            U_update=U_update,
                            verbose=verbose)
    del W_
    dims_ = list(map(np.shape,dW_))
    dW_ = combine_blocks(dims,
                        dW_,
                        list_order='C')

    if dx ==1:
        return dW_, rank_W_

    #########################
    # Row wise offset tiling
    #########################
    if verbose:
        print('Row wise tiling')
    W_rs, drs = offset_tiling(W,
                             nblocks=nblocks,
                             offset_case='r')


    #dims_=[dims,dims_rs,dims_cs,dims_rcs]
    #return W_,W_rs,W_cs,W_rcs, dims_

    dW_rs, rank_W_rs = run_single(W_rs,
                                confidence=confidence,
                                fudge_factor=fudge_factor,
                                greedy=greedy,
                                maxlag=maxlag,
                                mean_th_factor=mean_th_factor,
                                min_rank=min_rank,
                                snr_threshold=snr_threshold,
                                U_update=U_update,
                                verbose=verbose)
    del W_rs
    dims_rs = list(map(np.shape,dW_rs))

    dW_rs = combine_blocks(drs,
                        dW_rs,
                        list_order='C')

    #########################
    # Col wise offset tiling
    #########################
    if verbose:
        print('Col wise tiling')


    W_cs, dcs = offset_tiling(W,
                            nblocks=nblocks,
                            offset_case='c')

    dW_cs,rank_W_cs = run_single(W_cs,
                                confidence=confidence,
                                fudge_factor=fudge_factor,
                                greedy=greedy,
                                maxlag=maxlag,
                                mean_th_factor=mean_th_factor,
                                min_rank=min_rank,
                                snr_threshold=snr_threshold,
                                U_update=U_update,
                                verbose=verbose)
    del W_cs

    dims_cs = list(map(np.shape,dW_cs))

    dW_cs = combine_blocks(dcs,
                        dW_cs,
                        list_order='C')

    #########################
    # Row/Col wise offset tiling
    #########################
    if verbose:
        print('Row/Col wise tiling')


    W_rcs, drcs = offset_tiling(W,
                      nblocks=nblocks,
                      offset_case='rc')

    dW_rcs,rank_W_rcs = run_single(W_rcs,
                                  confidence=confidence,
                                  fudge_factor=fudge_factor,
                                  greedy=greedy,
                                  maxlag=maxlag,
                                  mean_th_factor=mean_th_factor,
                                  min_rank=min_rank,
                                  snr_threshold=snr_threshold,
                                  U_update=U_update,
                                  verbose=verbose)
    del W_rcs

    dims_rcs = list(map(np.shape,dW_rcs))

    dW_rcs = combine_blocks(drcs,
                            dW_rcs,
                            list_order='C')


    if False: # debug
        return nblocks, dW_, dW_rs, dW_cs, dW_rcs, dims_, dims_rs, dims_cs, dims_rcs

    W_four = combine_4xd(nblocks,
                         dW_,
                         dW_rs,
                         dW_cs,
                         dW_rcs,
                         dims_,
                         dims_rs,
                         dims_cs,
                         dims_rcs)

    return W_four , [rank_W_,rank_W_rs,rank_W_cs,rank_W_rcs]


def combine_4xd(nblocks,dW_,dW_rs,dW_cs,dW_rcs,dims_,dims_rs,dims_cs,dims_rcs,plot_en=False):
    """
    Inputs:
    -------


    Output:
    -------
    """
    dims = dW_.shape
    row_array,col_array = tile_grids(dims,
                                    nblocks=nblocks)

    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)

    r1, r2 = (row_array[1:]-r_offset)[[0,-1]]
    c1, c2 = (col_array[1:]-c_offset)[[0,-1]]

    drs     =   dW_rs.shape
    dcs     =   dW_cs.shape
    drcs    =   dW_rcs.shape

    # Get pyramid functions for each grid
    ak1 = np.zeros(dims[:2])
    ak2 = np.zeros(dims[:2])
    ak3 = np.zeros(dims[:2])

    ak0 = pyramid_tiles(dims,
                        dims_,
                        list_order='C')

    ak1[r1:r2,:] = pyramid_tiles(drs,
                               dims_rs,
                               list_order='C')

    ak2[:,c1:c2] = pyramid_tiles(dcs,
                                   dims_cs,
                                   list_order='C')


    ak3[r1:r2,c1:c2] = pyramid_tiles(drcs,
                                       dims_rcs,
                                       list_order='C')

    # Force outer most border = 1
    ak0[[0,-1],:]=1
    ak0[:,[0,-1]]=1

    #return ak0,ak1,ak2,ak3,patches,W_rs,W_cs,W_rcs
    if False:
        print('427 -- debug')
        return ak0,ak1,ak2,ak3

    W1 = np.zeros(dims)
    W2 = np.zeros(dims)
    W3 = np.zeros(dims)
    W1[r1:r2,:,:] = dW_rs
    W2[:,c1:c2,:] = dW_cs
    W3[r1:r2,c1:c2,:] = dW_rcs


    if plot_en:
        for ak_ in [ak0,ak1,ak2,ak3]:
            plt.figure(figsize=(10,10))
            plt.imshow(ak_[:,:])
            plt.show()

    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow((ak0+ak1+ak2+ak3)[:,:])
        plt.colorbar()

    W_hat = ak0[:,:,np.newaxis]*dW_
    W_hat += ak1[:,:,np.newaxis]*W1
    W_hat += ak2[:,:,np.newaxis]*W2
    W_hat += ak3[:,:,np.newaxis]*W3
    W_hat /= (ak0+ak1+ak2+ak3)[:,:,np.newaxis]
    return W_hat


def run_single(Y,
              confidence=0.99,
              debug = False,
              fudge_factor=1,
              greedy=False,
              maxlag=3,
              mean_th_factor=1.15,
              min_rank=1,
              parallel=True,
              snr_threshold=2,
              U_update=False,
              verbose=False
              ):
    """
    Run denoiser in each movie in the list Y.
    Inputs:
    ------
    Y:      list (number_movies,)
            list of 3D movies, each of dimensions (d1,d2,T)
            Each element in the list can be of different size.
    Outputs:
    --------
    Yds:    list (number_movies,)
            list of denoised 3D movies, each of same dimensions
            as the corresponding input movie.input
    vtids:  list (number_movies,)
            rank or final number of components stored for each movie.
    ------
    """
    if debug:
        print('485-debug')
        vtids = np.zeros((len(Y),))
        return Y, vtids

    mean_th = gpca.wnoise_acov_CI(Y[0].shape[2],
                             confidence=confidence,
                             maxlag=maxlag)
    if sys.platform == 'darwin':
        print('parallel version not for Darwin')
        parallel = False

    start = time.time()

    if parallel:
        cpu_count = max(1, multiprocessing.cpu_count()-2)
        args=[[patch] for patch in Y]
        start=time.time()
        pool = multiprocessing.Pool(cpu_count)
        print('Running %d blocks in %d cpus'%(len(Y),
                                              cpu_count))#if verbose else 0
        # define params in function
        c_outs = pool.starmap(partial(gpca.denoise_patch,
                              confidence=confidence,
                              fudge_factor=fudge_factor,
                              greedy=greedy,
                              maxlag=maxlag,
                              mean_th=mean_th,
                              mean_th_factor=mean_th_factor,
                              min_rank=min_rank,
                              snr_threshold=snr_threshold,
                              U_update=U_update,
                              verbose=verbose),
                              args)
        pool.close()
        pool.join()

        Yds = [out_[0] for out_ in c_outs]
        vtids = [out_[1] for out_ in c_outs]
    else:
        Yds = [None]*len(Y)
        vtids = [None]*len(Y)
        for ii, patch in enumerate(Y):
            print('Tile %d'%ii)
            #if not debug:
            y_ , vt_ = gpca.denoise_patch(patch,
                            confidence=confidence,
                            fudge_factor=fudge_factor,
                            greedy=greedy,
                            maxlag=maxlag,
                            mean_th=mean_th,
                            mean_th_factor=mean_th_factor,
                            min_rank=min_rank,
                            snr_threshold=snr_threshold,
                            U_update=U_update,
                            verbose=verbose)
            #else:
            #    y_ =patch
            #    vt_ = 0
            #print(vt_)
            Yds[ii] = y_
            vtids[ii] = vt_
    #print('535debug')
    #return
    vtids = np.asarray(vtids).astype('int')

    print('Blocks(=%d) run time: %f'%(len(Y),time.time()-start))
    return Yds, vtids


def run_single_deprecated_v2(Y,
               confidence=0.999,
               fudge_factor=0.99,
               greedy=False,
               maxlag=5,
               mean_th_factor=1.15,
               min_rank=1,
                parallel=True,
               U_update=False):
    """
    Run denoiser in each movie in the list Y.
    Inputs:
    ------
    Y:      list (number_movies,)
            list of 3D movies, each of dimensions (d1,d2,T)
            Each element in the list can be of different size.
    Outputs:
    --------
    Yds:    list (number_movies,)
            list of denoised 3D movies, each of same dimensions
            as the corresponding input movie.input
    vtids:  list (number_movies,)
            rank or final number of components stored for each movie.
    ------
    """

    def mp_worker(data_in,out_q):
        """ The worker function, invoked in a process
            'nums' is the input.
            The results are placed in a dictionary that's pushed to a queue.
        """
        outdict={}
        #print('Len is %d'%len(data_in))
        for ii, patch in enumerate(data_in):
            #print('Run for %d'%ii)
            #print(patch.shape)
            outdict[ii] = gpca.denoise_patch(patch,
                                  maxlag=maxlag,
                                  confidence=confidence,
                                  greedy=greedy,
                                  fudge_factor=fudge_factor,
                                  mean_th_factor=mean_th_factor,
                                  U_update=U_update,
                                  min_rank=min_rank,
                                  stim_knots=stim_knots,
                                  stim_delta=stim_delta)
            #print('out_q')
        out_q.put(outdict)

    # Each process will get 'chunksize' nums and a queue to put his out
    # dict
    # Parallel not for mac os single numpy default does not run with lapack
    if sys.platform == 'darwin':
        #print('Darwin')
        parallel = False

    start=time.time()
    print('debug')
    parallel =False
    if parallel:
        nprocs = max(1, multiprocessing.cpu_count()-2)
        out_q = multiprocessing.Queue()
        chunksize = int(ceil(len(Y) / float(nprocs)))
        procs = []

        for i in range(nprocs):
            p = multiprocessing.Process(
                    target=mp_worker,
                    args=(Y[chunksize * i:chunksize * (i + 1)],
                          out_q))
            procs.append(p)
            p.start()

        # Collect all results into a single result dict. We know how many dicts
        # with results to expect.
        resultdict = {}
        for i in range(nprocs):
            resultdict.update(out_q.get())

        # Wait for all worker processes to finish
        for p in procs:
            p.join()
        Yds=[]
        vtids=[]
        for c_out in resultdict:
            print(c_out)
            print(len(resultdict[c_out]))
            print(resultdict[c_out][0].shape)
            #for out_ in c_out:
            #    Yds.append(out_[0])
            #    vtids.append(out_[1])
        #print(len(Yds))
        #print(len(vtids))

            #Yds = #[out_[0] for out_ in c_out]
            #vtids = [out_[1] for out_ in c_out]
    else:
        Yds = [None]*len(Y)
        vtids = [None]*len(Y)
        for ii, patch in enumerate(Y):
            print('component %d'%ii)
            resultdict = gpca.denoise_patch(patch,
                                  maxlag=maxlag,
                                  confidence=confidence,
                                  greedy=greedy,
                                  fudge_factor=fudge_factor,
                                  mean_th_factor=mean_th_factor,
                                  U_update=U_update,
                                  min_rank=min_rank,
                                  stim_knots=stim_knots,
                                  stim_delta=stim_delta)
            Yds[ii]=resultdict[0]
            vtids[ii]=resultdict[1]

    vtids = np.asarray(vtids).astype('int')

    print('Run single video run time: %f'%(time.time()-start))
    return Yds, vtids


def run_single_deprecated(Y,
               confidence=0.999,
               fudge_factor=0.99,
               greedy=False,
               maxlag=5,
               mean_th_factor=1.15,
               min_rank=1,
               U_update=False,
               stim_knots=None,
               stim_delta=200):
    """
    Run denoiser in each movie in the list Y.
    GIL (Global Interpreter Lock) issues
    Inputs:
    ------
    Y:      list (number_movies,)
            list of 3D movies, each of dimensions (d1,d2,T)
            Each element in the list can be of different size.
    Outputs:
    --------
    Yds:    list (number_movies,)
            list of denoised 3D movies, each of same dimensions
            as the corresponding input movie.input
    vtids:  list (number_movies,)
            rank or final number of components stored for each movie.
    ------
    """
    args=[[patch] for patch in Y]
    cpu_count = 1#max(1, multiprocessing.cpu_count()-1)
    start=time.time()
    pool = multiprocessing.Pool(cpu_count)
    print('Running %d blocks in %d cpus'%(len(Y),cpu_count)) #if verbose else 0
    # define params in function
    c_outs = pool.starmap(partial(gpca.denoise_patch,
                                  maxlag=maxlag,
                                  confidence=confidence,
                                  greedy=greedy,
                                  fudge_factor=fudge_factor,
                                  mean_th_factor=mean_th_factor,
                                  U_update=U_update,
                                  min_rank=min_rank,
                                  stim_knots=stim_knots,
                                  stim_delta=stim_delta),
                                args)
    pool.close()
    pool.join()

    print('Run single video run time: %f'%(time.time()-start))
    Yds = [out_[0] for out_ in c_outs]
    vtids = [out_[1] for out_ in c_outs]
    vtids = np.asarray(vtids).astype('int')

    return Yds, vtids


def pyramid_matrix(dims,plot_en=False):
    """
    Compute a 2D pyramid function of size dims.

    Parameters:
    ----------
    dims:       tuple (d1,d2)
                size of pyramid function

    Outputs:
    -------
    a_k:        np.array (dims)
                 Pyramid function ranges [0,1],
                 where 0 indicates the boundary
                 and 1 the center.
    """
    a_k = np.zeros(dims[:2])
    xc, yc = ceil(dims[0]/2),ceil(dims[1]/2)

    for ii in range(xc):
        for jj in range(yc):
            a_k[ii,jj]=max(dims)-min(ii,jj)
            a_k[-ii-1,-jj-1]=a_k[ii,jj]
    for ii in range(xc,dims[0]):
        for jj in range(yc):
            a_k[ii,jj]=a_k[ii,-jj-1]
    for ii in range(xc):
        for jj in range(yc,dims[1]):
            a_k[ii,jj]=a_k[-ii-1,jj]
    a_k = a_k.max() - a_k
    a_k /=a_k.max()

    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow(a_k)
        plt.xticks(np.arange(dims[1]))
        plt.yticks(np.arange(dims[0]))
        plt.colorbar()
        plt.show()
    #if len(dims)>2:
        #a_k = np.array([a_k,]*dims[2]).transpose([1,2,0])
    return a_k


def pyramid_tiles(dims_rs,
                  dims_,
                  list_order='C',
                  plot_en=False):
    """
    Calculate 2D array of size dims_rs,
    composed of pyramid matrices, each of which has the same
    dimensions as an element in W_rs.
    Inputs:
    -------
    dims_rs:    tuple (d1,d2)
                dimension of array
    W_rs:       list
                list of pacthes which indicate dimensions
                of each pyramid function
    list_order: order in which the

    Outputs:
    --------

    """
    #dims_ = np.asarray(list(map(np.shape,W_rs)))
    a_ks = []
    for dim_ in dims_:
        a_k = pyramid_matrix(dim_)
        a_ks.append(a_k)
    # given W_rs and a_ks reconstruct array
    a_k = combine_blocks(dims_rs[:2],
                        a_ks,
                        dims_,
                        list_order=list_order)

    if plot_en:
        plt.figure(figsize=(10,10))
        plt.imshow(a_k)
        plt.colorbar()
    return a_k


def cn_ranks(dim_block, ranks, dims, list_order='C'):
    """
    """
    Crank = np.zeros(shape=dims)*np.nan
    d1,d2  = Crank.shape
    i,j = 0,0
    for ii in range(0,len(ranks)):
        d1c , d2c  = dim_block[ii][:2]
        Crank[i:i+d1c,j:j+d2c].fill(int(ranks[ii]))
        if list_order=='F':
            i += d1c
            if i == d1:
                j += d2c
                i = 0
        else:
            j+= d2c
            if j == d2:
                i+= d1c
                j = 0
    return Crank


def combine_blocks(dimsM,
                  Mc,
                  dimsMc=None,
                  list_order='C',
                  array_order='F'):
    """
    Combine blocks given by compress_blocks

    Parameters:
    ----------
    dimsM:          tuple (d1,d2,T)
                    dimensions of original array
    Mc:             np.array or list
                    contains (padded) tiles from array.
    dimsMc:         np.array of tuples (d1,d2,T)
                    (original) dimensions of each tile in array
    list_order:     string {'F','C'}
                    determine order to reshape tiles in array
                    array order if dxT instead of d1xd2xT assumes always array_order='F'
                    NOTE: if dimsMC is NONE then MC must be a d1 x d2 x T array
    array_order:    string{'F','C'}
                    array order to concatenate tiles
                    if Mc is (dxT), the outputs is converted to (d1xd2xT)
    Outputs:
    --------
    M_all:          np.array (dimsM)
                    reconstruction of array from Mc
    """

    ndims = len(dimsM)

    if ndims ==3:
        d1, d2, T = dimsM
        Mall = np.zeros(shape=(d1, d2, T))*np.nan
    elif ndims ==2:
        d1,d2 = dimsM[:2]
        Mall = np.zeros(shape=(d1, d2))*np.nan

    if type(Mc)==list:
        k = len(Mc)
    elif type(Mc)==np.ndarray:
        k = Mc.shape[0]
    else:
        print('error= must be np.array or list')
    if dimsMc is None:
        dimsMc = np.asarray(list(map(np.shape,Mc)))
    i, j = 0, 0
    for ii, Mn in enumerate(Mc):
        # shape of current block
        d1c, d2c = dimsMc[ii][:2]
        if (np.isnan(Mn).any()):
            Mn = unpad(Mn)
        if Mn.ndim < 3 and ndims ==3:
            Mn = Mn.reshape((d1c, d2c)+(T,), order=array_order)
        if ndims ==3:
            Mall[i:i+d1c, j:j+d2c, :] = Mn
        elif ndims ==2:
            Mall[i:i+d1c, j:j+d2c] = Mn
        if list_order=='F':
            i += d1c
            if i == d1:
                j += d2c
                i = 0
        else:
            j += d2c
            if j == d2:
                i += d1c
                j = 0
    return Mall


####################
# Deprecated
####################

def test_pyramids(dims,dims_rs,dims_cs,dims_rcs,W_1,W_rs,W_cs,W_rcs,row_cut,col_cut):
    """
    Input:
    ------
    Output:
    ------
    """
    ak0 = compute_ak(dims[:2],W_1,list_order='C')
    ak1 = compute_ak(dims_rs[:2],W_rs,list_order='F')
    ak2 = compute_ak(dims_cs[:2],W_cs,list_order='F')
    ak3 = compute_ak(dims_rcs[:2],W_rcs,list_order='F')
    ak0,ak1,ak2,ak3 = combine_4xd(dims,row_cut,col_cut,W_1,W_rs,W_cs,W_rcs)
    #plt.imshow((ak0+ak1+ak2+ak3)[:15,:15])
    for a_k in [ak0,ak1,ak2,ak3]:
        plt.figure(figsize=(15,10))
        plt.imshow((a_k)[:15,:15])
        plt.colorbar()

    plt.colorbar()
    print((ak0+ak1+ak2+ak3).min())
    np.argwhere((ak0+ak1+ak2+ak3)==0)
    return


def test_off_grids(mov_nn, nblocks=[10,10]):
    """
    Input:
    ------
    Output:
    ------
    """
    dims = mov_nn.shape

    ## denoiser 1
    W_ = split_image_into_blocks(mov_nn,nblocks=nblocks)

    dW_,rank_W_ = run_single(W_,debug=True)
    del W_
    dims_ = list(map(np.shape,dW_))
    dW_ = combine_blocks(dims,
                      dW_,
                      list_order='C')

    ## denoiser 2
    W_rs, drs = offset_tiling(mov_nn,
                           nblocks=nblocks,
                           offset_case='r')
    dW_rs,rank_W_rs = run_single(W_rs,debug=True)
    del W_rs
    dims_rs = list(map(np.shape,dW_rs))
    dW_rs = combine_blocks(drs,
                    dW_rs,
                    list_order='C')

    # denoiser 3
    W_cs, dcs = offset_tiling(mov_nn,
                        nblocks=nblocks,
                        offset_case='c')
    dW_cs,rank_W_cs = run_single(W_cs,
                              debug=True)
    del W_cs
    dims_cs = list(map(np.shape,dW_cs))
    dW_cs = combine_blocks(dcs,
                      dW_cs,
                      list_order='C')

    # denoiser 4
    W_rcs, drcs = offset_tiling(mov_nn,
                    nblocks=nblocks,
                    offset_case='rc')

    dW_rcs,rank_W_rcs = run_single(W_rcs,debug=True)
    del W_rcs
    dims_rcs = list(map(np.shape,dW_rcs))
    dW_rcs = combine_blocks(drcs,
                          dW_rcs,
                          list_order='C')
    row_array,col_array = tile_grids(dims,
                                  nblocks=nblocks)

    r_offset = vector_offset(row_array)
    c_offset = vector_offset(col_array)

    r1, r2 = (row_array[1:]-r_offset)[[0,-1]]
    c1, c2 = (col_array[1:]-c_offset)[[0,-1]]

    print (np.array_equiv(mov_nn,dW_))
    print (np.array_equiv(mov_nn[r1:r2,:,:],dW_rs))
    print (np.array_equiv(mov_nn[:,c1:c2,:],dW_cs))
    print (np.array_equiv(mov_nn[r1:r2,c1:c2,:],dW_rcs))
    return


def test_running_times(W,nblocks=[4,30]):
    """
    Input:
    ------
    Output:
    ------
    """
    dims = W.shape
    assert dims[2] >6000
    t_times = np.linspace(1000,7000,7).astype('int')
    run_times2= np.zeros((7,))
    for ii,ttime  in enumerate(t_times):
        start = time.time()
        _ = denoise_dx_tiles(image_[:,:,:ttime],nblocks=nblocks,dx=1)
        run_times2[ii]=time.time()-start
        print('Run for %f'%(run_times[ii]))


    plt.plot(t_times,run_times,'bo-')
    plt.xlabel('Number of [%d, %d] frames'%(dims[0],dims[1]))
    plt.ylabel('Run time [s]')
    return run_times


import numpy as np
import cv2
import scipy.ndimage.filters as filters
from scipy.ndimage.filters import convolve

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

#import caiman as cm
import trefide.extras.tools as tools_
import trefide.extras.tool_grid as tgrid

import trefide.extras.denoise as denoise
# update noise estimators
# clean up comments after revision

 
def correlation_pnr(Y,
                    gSig=None, #deprecated
                    center_psf=True,
                    remove_small_val =False,
                    remove_small_val_th =3
                   ):
                    #swap_dim=True):
    """
    compute the correlation image and the peak-to-noise ratio (PNR) image.
    If gSig is provided, then spatially filtered the video.

    Args:
        Y:  np.ndarray (3D or 4D).
            Input movie data in 3D or 4D format
        gSig:  scalar or vector.
            gaussian width. If gSig == None, no spatial filtering
        center_psf: Boolearn
            True indicates subtracting the mean of the filtering kernel
        swap_dim: Boolean
            True indicates that time is listed in the last axis of Y (matlab format)
            and moves it in the front

    Returns:
        cn: np.ndarray (2D or 3D).
            local correlation image of the spatially filtered (or not)
            data
        pnr: np.ndarray (2D or 3D).
            peak-to-noise ratios of all pixels/voxels

    """
    #if swap_dim:
    #    Y = np.transpose(
    #        Y, tuple(np.hstack((Y.ndim - 1,
    #            list(range(Y.ndim))[:-1]))))

    # parameters
    #_, d1, d2 = Y.shape


    #data_raw = Y.reshape(-1, d1, d2).astype('float32')

    # filter data
    #data_filtered = data_raw.copy()
    #if gSig:
    #    if not isinstance(gSig, list):
    #        gSig = [gSig, gSig]
    #    ksize = tuple([(3 * i) // 2 * 2 + 1 for i in gSig])
        # create a spatial filter for removing background
        # psf = gen_filter_kernel(width=ksize, sigma=gSig, center=center_psf)

    #    if center_psf:
    #        for idx, img in enumerate(data_filtered):
    #            data_filtered[idx, ] = cv2.GaussianBlur(img,
    #                                                    ksize=ksize,
    #                                                    sigmaX=gSig[0],
    #                                                    sigmaY=gSig[1],
    #                                                    borderType=1) \
    #                - cv2.boxFilter(img, ddepth=-1, ksize=ksize, borderType=1)
    #        # data_filtered[idx, ] = cv2.filter2D(img, -1, psf, borderType=1)
    #    else:
    #        for idx, img in enumerate(data_filtered):
    #            data_filtered[idx, ] = cv2.GaussianBlur(
    #                img, ksize=ksize, sigmaX=gSig[0], sigmaY=gSig[1], borderType=1)

    # compute peak-to-noise ratio
    #data_filtered -= np.mean(data_filtered, axis=0)
    Y = Y - Y.mean(2,keepdims=True)
    #data_max = np.max(data_filtered, axis=0)
    data_max = Y.max(2)#,keepdims=True)
    #data_std = noise_estimator.get_noise_fft(data_filtered.transpose())[0].transpose()
    #data_std = noise_estimator.get_noise_fft(Y)[0]
    data_std = denoise.noise_level(Y)#[0]
    # Update to match noise from denoise.py here

    ## data_std = get_noise(data_filtered, method='diff2_med')
    pnr = np.divide(data_max, data_std)
    if remove_small_val:
        pnr[pnr < 0] = 0

    tmp_data = Y / data_std[:,:,np.newaxis]
    # remove small values
    #tmp_data = data_filtered.copy() / data_std
    if remove_small_val:
        tmp_data[tmp_data < remove_small_val_th] = 0

    # compute correlation image
    # cn = local_correlation(tmp_data, d1=d1, d2=d2)
    #cn = local_correlations_fft(tmp_data, swap_dim=False)
    cn = local_correlations_fft(tmp_data, swap_dim=True)

    return cn, pnr


def local_correlations_fft(Y,
                            eight_neighbours=True,
                            swap_dim=True,
                            opencv=True):
    """Computes the correlation image for the input dataset Y using a faster FFT based method

    Parameters:
    -----------

    Y:  np.ndarray (3D or 4D)
        Input movie data in 3D or 4D format

    eight_neighbours: Boolean
        Use 8 neighbors if true, and 4 if false for 3D data (default = True)
        Use 6 neighbors for 4D data, irrespectively

    swap_dim: Boolean
        True indicates that time is listed in the last axis of Y (matlab format)
        and moves it in the front

    opencv: Boolean
        If True process using open cv method

    Returns:
    --------

    Cn: d1 x d2 [x d3] matrix, cross-correlation with adjacent pixels

    """

    if swap_dim:
        Y = np.transpose(
            Y, tuple(np.hstack((Y.ndim - 1, list(range(Y.ndim))[:-1]))))

    Y = Y.astype('float32')
    Y -= np.mean(Y, axis=0)
    Ystd = np.std(Y, axis=0)
    Ystd[Ystd == 0] = np.inf
    Y /= Ystd

    if Y.ndim == 4:
        if eight_neighbours:
            sz = np.ones((3, 3, 3), dtype='float32')
            sz[1, 1, 1] = 0
        else:
            sz = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                           [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]], dtype='float32')
    else:
        if eight_neighbours:
            sz = np.ones((3, 3), dtype='float32')
            sz[1, 1] = 0
        else:
            sz = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype='float32')

    if opencv and Y.ndim == 3:
        Yconv = Y.copy()
        for idx, img in enumerate(Yconv):
            Yconv[idx] = cv2.filter2D(img, -1, sz, borderType=0)
        MASK = cv2.filter2D(
            np.ones(Y.shape[1:], dtype='float32'), -1, sz, borderType=0)
    else:
        Yconv = convolve(Y, sz[np.newaxis, :], mode='constant')
        MASK = convolve(
            np.ones(Y.shape[1:], dtype='float32'), sz, mode='constant')
    Cn = np.mean(Yconv * Y, axis=0) / MASK
    return Cn


def cn_ranks_dx_plot(ranks,
                  dims,
                  nblocks=[10, 10],
                    figsize=15,
                    fontsize=20,
                    tile_err=100,
                    include_err=True,
                    save_fig=False,
                   save_fig_name=''):
    rtype=[None,'r','c','rc']
    for ii,rank in enumerate(ranks):
        cname_= save_fig_name+'_offset_'+str(rtype[ii])+'_'
        if not include_err:
            rank=rank%tile_err
            rank[rank==0]=1

        cn_ranks_plot(rank,
                  dims,
                  nblocks=nblocks,
                  offset_case=rtype[ii],
                     figsize=figsize,
                     fontsize=fontsize,
                     save_fig=save_fig,
                     save_fig_name=cname_)
    return


def cn_ranks_plot(ranks,
                  dims,
                  nblocks=[10, 10],
                  offset_case=None,
                  list_order='C',
                  exclude_max=True,
                  max_rank=100,
                 fontsize=20,
                 figsize=15,
                 save_fig_name='',
                 save_fig=False):
    """
    Plot rank array given ranks of individual tiles,
    and tile coordinates.

    Parameters:
    ----------
    dim_block:
    ranks:
    dims:

    Outputs:
    -------
    Cplot3:         np.array
                    array of ranks per tile
    """

     #offset_tiling_dims(dims,nblocks,offset_case=None):

    dims, dim_block = tgrid.offset_tiling_dims(dims,
                                               nblocks,
                                               offset_case=offset_case)


    K1 = nblocks[0] - 1
    K2 = nblocks[1] - 1

    K1 = K1-1 if offset_case =='r' else K1
    K2 =K2-1 if offset_case =='c' else K2
    K1 =K1-1 if offset_case =='rc' else K1
    K2 =K2-1 if offset_case =='rc' else K2


    Cplot3 = tgrid.cn_ranks(dim_block, ranks,
                            dims[:2], list_order=list_order)
    d1, d2 = dims[:2] // np.min(dims[:2])
    fig, ax3 = plt.subplots(1, 1, figsize=(d1 * figsize, d2 * figsize))

    ranks_ = ranks.copy()
    if exclude_max:
        ranks_[ranks > max_rank] = ranks[ranks > max_rank] % max_rank

    ranks_std = np.std(ranks_)
    vmin_ = max(0, ranks_.min() - ranks_std)
    vmax_ = ranks_.max() + ranks_std

    ax3.set_title('Ranks in each tile %d' % (
        np.sum(np.asarray(ranks_))))
    im3 = ax3.imshow(Cplot3, vmin=vmin_,
                     vmax=vmax_, cmap='Reds',
                     interpolation='nearest', aspect='equal')

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="2%", pad=0.05)
    plt.colorbar(im3, cax=cax3, format='%d',
                 ticks=np.linspace(vmin_, vmax_, 5))

    dim_block = np.asarray(dim_block)
    cols, rows = dim_block.T[0], dim_block.T[1]


    row_array = np.insert(rows[::K1 + 1], 0, 0).cumsum()
    col_array = np.insert(cols[::K2 + 1], 0, 0).cumsum()

    x, y = np.meshgrid(row_array[:-1],
                        col_array[:-1])
    ax3.set_yticks(col_array[:-1])
    ax3.set_xticks(row_array[:-1])

    for ii, (row_val, col_val) in enumerate(zip(x.flatten(order=list_order),
                                                y.flatten(order=list_order))):
        c = str(int(Cplot3[int(col_val + 1), int(row_val + 1)]) % max_rank)
        ax3.text(row_val + rows[ii] / 2, col_val +
                 cols[ii] / 2, c, va='center', ha='center',fontsize=fontsize)
    plt.tight_layout()
    if save_fig:
        save_fig_name = save_fig_name+'ranks_plot.pdf'
        plt.savefig(save_fig_name)
    else:
        plt.show()

    return Cplot3


def plot_comp(Y, Y_hat=None, title_=None, dims=None, idx_=0):
    """
    Plot comparison for frame idx_ in Y, Y_hat.
    assume Y is in dxT to be reshaped to dims=(d1,d2,T)
    """
    if Y_hat is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
        ax.set_title(title_)
        plots_ = zip([ax], [Y])
    else:
        R = Y - Y_hat
        fig, ax = plt.subplots(1, 3, figsize=(15, 6))
        ax[0].set_title(title_)
        plots_ = zip(ax, [Y, Y_hat, R])

    for ax_, arr in plots_:
        if np.ndim(arr) > 2:
            ims = ax_.imshow(arr.reshape(dims, order='F')[:, :, idx_])
        else:
            ims = ax_.imshow(arr.reshape(dims[:2], order='F'))
        #ims = ax_.imshow(arr.reshape(dims,order='F').var(2))
        d = make_axes_locatable(ax_)
        cax0 = d.append_axes("bottom", size="5%", pad=0.5)
        cbar0 = plt.colorbar(
            ims, cax=cax0, orientation='horizontal', format='%.0e')
    plt.tight_layout()
    plt.show()
    return


def plot_temporal_traces(V_TF, V_hat=None,title_=''):
    """
    """
    if np.ndim (V_TF)==1:
        V_TF =V_TF[np.newaxis,:]
    if np.ndim (V_hat)==1:
        v_hat = V_hat[np.newaxis,:]
    else:
        v_hat = V_hat

    for idx, vt in enumerate(np.asarray(V_TF)):
        plt.figure(figsize=(15, 5))
        plt.title(('Temporal component %d'+ title_)% idx)
        plt.plot(vt, 'b-')
        if v_hat is not None:
            plt.plot(v_hat[idx, :], 'r--')
            plt.legend(['raw','denoised'])
        plt.show()
    return




def plot_spatial_component(U_, Y_hat=None,dims=None):
    """
    """
    if np.ndim(U_) ==1:
        U_=U_[:,np.newaxis]

    if np.ndim(Y_hat) ==1:
        Y_hat=Y_hat[:,np.newaxis]

    U_hat_c = None
    for ii in range(U_.shape[1]):
        if Y_hat is not None:
            U_hat_c = Y_hat[:,ii]
        plot_comp(U_[:, ii],
            Y_hat=U_hat_c,
            title_='Spatial component U' +
                  str(ii), dims=dims[:2])
    return


def plot_vt_cov(Vt1, keep1, maxlag):
    """
    Plot figures of ACF of vectors in Vt1 until maxlag
    (right pannel) keep1 and (left pannel) other components

    Parameters:
    ----------

    Vt1:    np.array (k xT)
            array of k temporal components lasting T samples
    keep1: np.array (k1 x 1)
            array of temporal components which passed a given hypothesis
    maxlag: int
            determined lag until which to plot ACF function of row-vectors
            in Vt1

    Outputs:
    -------

    None:   displays a figure

    """
    fig, axarr = plt.subplots(1, 2, sharey=True,figsize=(10,5))
    loose = np.setdiff1d(np.arange(Vt1.shape[0]), keep1)
    for keep in keep1:
        vi = Vt1[keep, :]
        vi = (vi - vi.mean()) / vi.std()
        metric = tools_.axcov(vi, maxlag)[maxlag:] / vi.var()
        axarr[0].plot(metric, '-r',linewidth=1.0)

    for lost in loose:
        vi = Vt1[lost, :]
        vi = (vi - vi.mean()) / vi.std()
        metric = tools_.axcov(vi, maxlag)[maxlag:] / vi.var()
        axarr[1].plot(metric, ':b',linewidth=2.0)

    ttext = ['Selected components: %d' % (len(keep1)),
             'Discarded components: %d' % (len(loose))]
    for ii, ax in enumerate(axarr):
        ax.set_xscale('symlog')
        ax.set_ylabel('ACF')
        ax.set_xlabel('lag')
        ax.set_yticks([0,0.5,1])
        ax.set_title(ttext[ii])
    #plt.savefig('cosyne_Vt_keep.svg')
    plt.show()
    return


def show_img(img,
             ax=None,
             vmin=None,
             vmax=None,
             cbar_orientation='horizontal',
             plot_colormap='jet',
             plot_size=(12,7),
             cbar_ticks_number=None,
             cbar_ticks=None,
             cbar_enable=True):
    """
    Visualize image
    """

    if ax is None:
        fig = plt.figure(figsize=plot_size)
        ax = plt.subplot(111)

    vmin= img.min() if vmin is None else vmin
    vmax= img.max() if vmax is None else vmax


    if np.abs(img.min()) <= 1.5:
        if np.abs(img.min()) <= -1e-1:
            format_tile = '%.1e'
        else:
            format_tile = '%.2f'
    else:
        format_tile = '%5d'

    if cbar_ticks_number is not None:
        cbar_ticks= np.linspace(vmin,
                                vmax,
                                cbar_ticks_number,
                                endpoint=True)
        cbar_ticks=np.round(cbar_ticks,4)
        cbar_ticks_labels= [format_tile%(cbar_)
                            for cbar_ in cbar_ticks]
        vmin, vmax = cbar_ticks[0], cbar_ticks[-1]

    #######################
    # Build Plot
    #######################

    d1,d2= img.shape
    im = ax.imshow(img,
                   vmin=vmin,
                   vmax=vmax,
                   cmap=plot_colormap,
                   extent=[0,d2,0,d1])

    divider = make_axes_locatable(ax)
    if cbar_orientation == 'horizontal':
        cbar_direction ='bottom'
    elif cbar_orientation == 'vertical':
        cbar_direction ='right'

    if cbar_enable is False:
        return
    cax = divider.append_axes(cbar_direction,
                          size="5%",
                          pad=0.3)

    cbar = plt.colorbar(im,
                        cax=cax,
                        orientation=cbar_orientation,
                        spacing='uniform',
                        format=format_tile,
                        ticks=cbar_ticks)
    return


def nearest_frame_corr(A):
    """
    """
    num_frames = A.shape[2]
    corrs = np.zeros((num_frames-1,))
    for idx in range(num_frames-1):
        frame1 = A[:,:,idx].flatten()
        frame2 = A[:,:,idx+1].flatten()
        corrs[idx] =  corr(frame1,frame2)
    return corrs


def corr(a,b):
    a -= a.mean()
    b -= b.mean()
    return a.dot(b) / sqrt(a.dot(a) * b.dot(b) + np.finfo(float).eps)


def correlation_traces(Y,Yd,R):
    """
    we also talked about computing "correlation traces"
    instead of correlation images - ie, instead of computing
    correlations between nearest neighbor pixels and showing
    these as an image, compute correlations between nearest
    neighbor frames and show these as a trace.  let's try this too.
    lots of ways to slice and dice things to look at
    how much signal we're losing / check to see that
    we're just leaving noise behind.
    """

    corrs_Y = nearest_frame_corr(Y)
    corrs_Yd = nearest_frame_corr(Yd)
    corrs_R = nearest_frame_corr(R)

    plt.figure(figsize=(10,10))
    plt.title('Correlation traces')
    plt.plot(corrs_Y)
    plt.plot(corrs_Yda)
    plt.plot(corrs_R)
    plt.legend(['raw','denoised','residual'])

    return 0

def snr_per_frame(Y,Yd,R,
                    cbar_orientation='vertical',
                    plot_orientation='horizontal',
                    title=True,
                    titles_=['SNR_frame']):
    """
    take a patch and sum Y, Yd, and R over all pixels to get three traces.
    (or even just do this on a single pixel -
    this would be analogous to looking at a sample frame from the movie.)
    the Yd trace will presumably look a lot like the
    Y trace (maybe with slightly less noise) and hopefully the R trace
    will just look like noise.

    for the paper we'll also want to combine most (maybe all)
    of these panels into one big fig - can you do this too?
    """
    # given patch sum over all pixels
    titles_ = titles_*3
    titles_[0].append(' raw')
    titles_[1].append(' denoised')
    titles_[2].append(' residual')

    Ys = Y.sum(2)
    Yds = Yd.sum(2)
    Rs = R.sum(2)
    comparison_plot([Ys,Yds,Rs],
                    cbar_orientation=cbar_orientation,
                    option='input',
                    plot_orientation=plot_orientation,
                    title=title,
                    titles_=titles_)
    return



def nearest_frame_corr(A):
    """
    """
    num_frames = A.shape[2]
    corrs = np.zeros((num_frames-1,))
    for idx in range(num_frames-1):
        frame1 = A[:,:,idx].flatten()
        frame2 = A[:,:,idx+1].flatten()
        corrs[idx] =  corr(frame1,frame2)
    return corrs


def corr(a,b):
    a -= a.mean()
    b -= b.mean()
    return a.dot(b) / np.sqrt(a.dot(a) * b.dot(b) +np.finfo(float).eps)


def correlation_traces(Y,Yd,R):
    """
    we also talked about computing "correlation traces"
    instead of correlation images - ie, instead of computing
    correlations between nearest neighbor pixels and showing
    these as an image, compute correlations between nearest
    neighbor frames and show these as a trace.  let's try this too.
    lots of ways to slice and dice things to look at
    how much signal we're losing / check to see that
    we're just leaving noise behind.
    """

    corrs_Y = nearest_frame_corr(Y)
    corrs_Yd = nearest_frame_corr(Yd)
    corrs_R = nearest_frame_corr(R)

    plt.figure(figsize=(10,6))
    plt.title('Correlation traces')
    plt.plot(corrs_Y)
    plt.plot(corrs_Yd)
    plt.plot(corrs_R)
    plt.legend(['raw','denoised','residual'])

    return 0


def comparison_plot(cn_see,
                    option='corr',
                    plot_orientation='horizontal',
                    cbar_orientation='vertical',
                    cbar_indiv_range=None,
                    title=True,
                    title_suffix='',
                    titles_='',
                    share_colorbar=False,
                    plot_colormap='jet',
                    plot_num_samples=1000,
                    remove_small_val_th=3,
                    remove_small_val=False,
                    plot_size = 12,
                    cbar_ticks_number=None,
                   save_fig=False,
                   save_fig_name='corr_'):
    """
    """
    num_plots = len(cn_see)

    if share_colorbar:
        min_dim=4
    else:
        min_dim=3

    if titles_=='' and num_plots==2:
        titles_=['original ','denoised ']

    if num_plots==2:
        cn_see.append(cn_see[0]-cn_see[1])
        titles_.append('residual ')

    if plot_orientation == 'horizontal':
        d1, d2 = min_dim,1
        sharex = True
        sharey = False

    elif plot_orientation =='vertical':
        d1, d2 = 1,min_dim
        sharex = False
        sharey = True

    Cn_all =[]

    #######################
    # Calculate Cn to plot
    #######################
    for ii, array in enumerate(cn_see):
        #print(array.shape)
        if option =='corr': # Correlation
            Cn, _ = correlation_pnr(array,
                                    gSig=None,
                                    remove_small_val=remove_small_val,
                                    remove_small_val_th=remove_small_val_th,
                                    center_psf=False)#,
                                    #swap_dim=True) # 10 no ds

            title_prefix = 'Local correlation: '
        elif option =='var': #Variance
            Cn = array.var(2)/array.shape[2]
            title_prefix = 'Pixel variance: '
            #print(Cn.min())
            #print(Cn.max())
        elif option =='pnr': # PNR
            _, Cn = correlation_pnr(array,
                                    gSig=None,
                                    remove_small_val=remove_small_val,
                                    remove_small_val_th=remove_small_val_th,
                                    center_psf=False)#,
                                    #swap_dim=True)
            title_prefix = 'PNR: '

        elif option=='input':
            Cn =array - array.min()
            Cn = Cn/Cn.max()
            title_prefix = 'Single Frame: '

        elif option=='snr':
            Cn1 = array.std(2)
            Cn2 = denoise.noise_level(array)
            Cn = Cn1/Cn2
            title_prefix = 'SNR: '
        else:
            title_prefix = ''


        print ('%s range [%.1e %.1e]'%(title_prefix,
                                   Cn.min(),
                                   Cn.max()))
        Cn_all.append(Cn)

    #######################
    # Plot configuration
    #######################
    vmax_ = list(map(np.max,Cn_all))
    vmin_ = list(map(np.min,Cn_all))

    if share_colorbar:
        vmax_ = [max(vmax_)]*3
        vmin_ = [min(vmin_)]*3

    if cbar_indiv_range is not None:
        for ii,range_ in enumerate(cbar_indiv_range):
            vmin_[ii]=range_[0]
            vmax_[ii]=range_[1]

    dim2, dim1 = Cn.shape
    x_ticks= np.linspace(0,dim1,5).astype('int')
    y_ticks= np.linspace(0,dim2,5).astype('int')

    fig, axarr = plt.subplots(d1,d2,
                              figsize=(d1*plot_size,d2*plot_size),
                              sharex=sharex,
                              sharey=sharey)

    #cbar_enable= [False,False,True]
    cbar_enable= not share_colorbar


    for ii, Cn in enumerate(Cn_all):
        show_img(Cn,
                 ax =axarr[ii],
                 cbar_orientation=cbar_orientation,
                 vmin=vmin_[ii],
                 vmax=vmax_[ii],
                 plot_colormap=plot_colormap,
                 cbar_ticks_number=cbar_ticks_number,
                 cbar_enable=cbar_enable)

        axarr[ii].set_xticks(x_ticks)
        axarr[ii].set_yticks(y_ticks)
        axarr[ii].set_xticklabels([])
        axarr[ii].set_yticklabels([])

        if title:
            axarr[ii].set_title(title_prefix
                                + titles_[ii]
                                + title_suffix)


    plt.tight_layout()
    if save_fig:
        save_fig_name = save_fig_name+'comparison_plot_'+'.pdf'
        plt.savefig(save_fig_name)
    else:
        plt.show()
    return


def intialization_plot(data_highpass,
                       patch_radius=20,
                       min_pnr=0,
                       min_corr=0,
                       stdv_pixel=None,
                       noise_thresh=3,
                       orientation='horizontal'):  # down,side
    """
    """

    # Create plot obj according to specifications
    if orientation == 'horizontal':
        d1, d2 = 2, 1
    elif orientation == 'vertical':
        d1, d2 = 1, 2
    fig, axarr = plt.subplots(d1, d2, figsize=(14, 7), sharex=True)

    # Compute pixel-wise noise stdv
    if not stdv_pixel:
        stdv_pixel = np.sqrt(np.var(data_highpass, axis=-1))

    # Compute & plot corr image
    data_spikes = data_highpass - \
        np.median(data_highpass, axis=-1)[:, :, np.newaxis]
    data_spikes[data_spikes < noise_thresh * stdv_pixel[:, :, np.newaxis]] = 0
    corr_image = local_correlations_fft(
        data_spikes.transpose([2, 0, 1]), swap_dim=False)

    if min_corr:
        corr_image[corr_image < min_corr] = 0
    show_img(axarr[0], corr_image, orientation=orientation)
    axarr[0].set_title('Thresholded Corr Image')

    # Compute & plot pnr image
    pnr_image = np.divide(np.max(data_highpass, axis=-1),
                          stdv_pixel)
    pnr_image[np.logical_or(corr_image < min_corr, pnr_image < min_pnr)] = 0
    pnr_image = filters.median_filter(pnr_image,
                                      size=(int(round(patch_radius / 4)),) * 2,
                                      mode='constant')
    show_img(axarr[1], pnr_image, orientation=orientation)
    axarr[1].set_title('Thresholded & Filtered PNR Image')

    # Display PLot
    plt.tight_layout()
    plt.show()
    return


def tiling_grid_plot(W,
                     nblocks=[10, 10],
                    plot_option='var'):
    """
    """
    dims = W.shape
    col_array, row_array = tgrid.tile_grids(dims,
                                        nblocks=nblocks)
    x, y = np.meshgrid(row_array, col_array)
    if plot_option == 'var':
        Cn1 = W.var(2)
    elif plot_option =='same':
        Cn1 = W
    plt.figure(figsize=(15, 5))
    plt.yticks(col_array)
    plt.xticks(row_array)
    plt.plot(x.T, y.T)
    plt.plot(x, y)
    plt.imshow(Cn1)
    plt.show()
    return


def spatial_filter_spixel_plot(data,y_hat,hat_k):
    Cn_y, _ = correlation_pnr(data) #
    Cn_yh,_ = correlation_pnr(y_hat)

    fig,ax = plt.subplots(1,3,figsize=(10,5))
    im0 = ax[0].imshow(Cn_y.T,vmin=maps[0],vmax=maps[1])
    if neuron_indx is None:
        im1 = ax[1].imshow(hat_k)
    else:
        im1 = ax[1].imshow(hat_k[:,np.newaxis].T)
    im2 = ax[2].imshow(Cn_yh.T,vmin=maps[0],vmax=maps[1])
    ax[0].set_title('y')
    ax[1].set_title('k')
    ax[2].set_title('y_hat')

    ax[0].set_xticks(np.arange(y_hat.shape[0]))
    ax[0].set_yticks(np.arange(y_hat.shape[1]))
    ax[2].set_xticks(np.arange(y_hat.shape[0]))
    ax[2].set_yticks(np.arange(y_hat.shape[1]))
    ax[1].set_yticks(np.arange(1))

    if neuron_indx is None:
        ax[1].set_xticks(np.arange(np.prod(y_hat.shape[:2]))[::4])
        ax[1].set_yticks(np.arange(np.prod(y_hat.shape[:2]))[::4])

    divider0 = make_axes_locatable(ax[0])
    cax0 = divider0.append_axes("bottom", size="5%", pad=0.5)
    cbar0 = plt.colorbar(im0, cax=cax0, orientation='horizontal')
    divider1 = make_axes_locatable(ax[1])
    cax1 = divider1.append_axes("bottom", size="5%", pad=0.5)
    cbar1 = plt.colorbar(im1, cax=cax1, format="%.2f", orientation='horizontal')
    divider2 = make_axes_locatable(ax[2])
    cax2 = divider2.append_axes("bottom", size="5%", pad=0.5)
    cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')
    plt.tight_layout()
    plt.show()
    return

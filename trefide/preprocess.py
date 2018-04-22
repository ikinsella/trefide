import scipy

import numpy as np
import scipy.interpolate as interp
import scipy.ndimage.filters as filt
import matplotlib.pyplot as plt


def flag_outliers(signal,
                  thresh_stdv=4,
                  buffer=10,
                  visualize=False):
    """ Flag outliers based on median abs deviation.
        Returns two arrays of indices.
        The first gives the indices to be deleted.
        The second gives the indices of locations in the new signal which
        will potentially have discontinuities due to fluroescence reset.
    """

    # z-score to locate outliers
    keep_idx = abs(signal - np.median(signal)) < thresh_stdv * np.std(signal)

    # minimum filter removes pixels within buffer distance of outliers
    keep_idx = filt.minimum_filter(keep_idx, size=2 * buffer + 1)

    # Plot flagged outliers -- hacky so may break if params unreasonable
    if visualize:
        fig = plt.figure(figsize=(16, 4))

        trans_idx = np.argwhere(filt.convolve1d(keep_idx, np.array([1, -1])))
        for idx in range(len(trans_idx)):
            if idx == 0:
                plt_idx = np.arange(0, trans_idx[idx])
            else:
                plt_idx = np.arange(trans_idx[idx - 1], trans_idx[idx])

            color = 'b' if keep_idx[trans_idx[idx] - 1] else 'r'
            plt.plot(plt_idx, signal[plt_idx], color)

        if trans_idx[-1] < len(signal):
            plt_idx = np.arange(trans_idx[idx], len(signal))
            color = 'b' if keep_idx[len(signal) - 1] else 'r'
            plt.plot(plt_idx, signal[plt_idx], color)

        plt.plot(np.arange(len(signal)),
                 (np.ones(len(signal)) * np.median(signal)) -
                 (thresh_stdv * np.std(signal)), 'g')
        plt.plot(np.arange(len(signal)),
                 (np.ones(len(signal)) * np.median(signal)) +
                 (thresh_stdv * np.std(signal)), 'g')
        plt.title('Outliers Flagged For Removal & Threshold')
        plt.show()

    # List of indices to be deleted
    del_idx = np.argwhere(~keep_idx)

    # list of indices where samples were cutout (possible discontinuities)
    disc_idx = np.argwhere(filt.convolve1d(
        keep_idx, np.array([1, -1]))[keep_idx])

    return del_idx, disc_idx


def _get_knots(stim,
               k=3,
               followup=100,
               spacing=250):

    # Locate transition indices
    trans_idx = np.argwhere(filt.convolve1d(stim > 0, np.array([1, -1])))
    # Repeat knots and add transition extras
    knots = np.append(np.append(np.zeros(k + 1),
                                np.sort(np.append(np.repeat(trans_idx, k),
                                                  trans_idx + followup))),
                      np.ones(k + 1) * len(stim)).astype('int')

    # add regularly spaced extra knots between transitions
    extras = np.empty(0)

    for idx in np.linspace(k + 1,
                           len(knots),
                           int(np.ceil(len(knots) / (k + 1))), dtype='int')[:-1]:
        extras = np.append(
            extras,
            np.linspace(knots[idx - 1], knots[idx],
                        int(np.round(
                            (knots[idx] - knots[idx - 1]) / spacing)) + 2,
                        dtype='int')[1:-1]
        )

    # Locate beginning/end of transition zones as knots
    return np.sort(np.append(knots, extras)).astype('int')


def _get_spline_trend(data,
                      stim,
                      order=3,
                      followup=100,
                      spacing=200,
                      q=.05,
                      axis=-1,
                      robust=True,
                      disc_idx=None):
    """
    Fits an adaptive b-spline to an input dataset in order to remove slow
    trend and features due to application of step and ramp stimuli.
    TODO: docs
    """
    # get knots from stim
    knots = _get_knots(stim, k=order, followup=100, spacing=250)
    x = np.arange(len(stim))

    if disc_idx is not None:
        knots = np.sort(np.append(knots, np.repeat(disc_idx, order + 1)))

    def spline_fit(y):
        bspl = interp.make_lsq_spline(x=x, y=y, t=knots, k=order)
        return bspl(x)

    def robust_spline_fit(y):
        bspl = interp.make_lsq_spline(x=x, y=y, t=knots, k=order)
        resid = np.abs(bspl(x) - y)
        keep_idx = resid <= np.percentile(resid, (1 - q) * 100)
        bspl = interp.make_lsq_spline(
            x=x[keep_idx], y=y[keep_idx], t=knots, k=order)

        return bspl(x)

    # fit spline To whole dataset
    if robust:
        trend = np.apply_along_axis(robust_spline_fit, axis, data)
    else:
        trend = np.apply_along_axis(spline_fit, axis, data)

    return trend


def detrend(mov,
            stim,
            disc_idx,
            order=3,
            followup=100,
            spacing=200,
            q=.05,
            axis=-1,
            robust=True,
            visualize=None):
    """ Detrends Q-state video via stim & discontinuity spline fit. 
    Removes potential discontinuity artifacts afterwards
    TODO: docs
    """
    # Adaptive spline fit
    trend = _get_spline_trend(data=mov,
                              stim=stim,
                              disc_idx=disc_idx,
                              order=order,
                              followup=followup,
                              spacing=spacing,
                              q=q,
                              axis=axis,
                              robust=robust)

    # Remove samples from discontinuity locations
    del_idx = np.sort(np.append(np.append(disc_idx, disc_idx + 1),
                                disc_idx - 1))
    stim = np.delete(stim, del_idx)
    mov_detr = np.delete(np.subtract(mov, trend), del_idx, axis=-1)
    trend = np.delete(trend, del_idx, axis=-1)

    # Optionally show spline fit to single pixel
    if visualize:
        row = visualize[0]
        col = visualize[1]
        T = len(stim)
        f, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        ax1.plot(np.arange(T), np.delete(mov[row, col, :], del_idx), 'b')
        ax1.plot(np.arange(T), trend[row, col, :], 'r')
        ax1.set_title('Raw Pixel and Spline Fit')
        ax2.plot(np.arange(T), mov_detr[row, col, :], 'b')
        ax2.set_title('Detrended Pixel')
        plt.show()

    # Recompute problem areas
    disc_idx[1:] = disc_idx[1:] - np.cumsum(np.ones(len(disc_idx) - 1) * 3)
    disc_idx = disc_idx - 1
    disc_idx = np.append(disc_idx,
                         np.argwhere(filt.convolve1d(stim > 0,
                                                     np.array([1, -1]))))
    return mov_detr, trend, stim, np.unique(disc_idx)


def retrend(trend_components,
            disc_idx,
            stim,
            all_quad=False):
    """ Refit the raw data with trend after removing photobleach trend from 
        each stim onset
    """
    bleach_trend, del_idx = _get_photobleach_trend(trend_components,
                                                   disc_idx,
                                                   stim,
                                                   all_quad=all_quad)

    # final_trend = np.delete(temporal_components +
    #                         trend_components - bleach_trend, del_idx, axis=-1)
    # final_stim = np.delete(stim, del_idx)
    stim_trend = trend_components - bleach_trend
    plot_idx = np.ones(len(stim), dtype='bool')
    plot_idx[del_idx] = False
    return stim_trend, plot_idx  # del_idx


def _get_photobleach_trend(trend_components, disc_idx, stim, all_quad=False):
    """ Fit trend to samples where stim was off in each segment to remove
        photobleach related fluorescence decay
    """
    disc_idx = np.setdiff1d(disc_idx, np.argwhere(
        scipy.ndimage.filters.convolve(stim > 0, np.array([1, -1]))))
    stim_off = stim <= 0
    bleach_trend = np.zeros(trend_components.shape)

    # Fit each recorded segment separately (as defined by artifacts removed)
    for n in range(len(disc_idx) + 1):

        # Index active section
        if n == 0:
            signals = trend_components[:, :disc_idx[n]]
            stim_off_idx = stim_off[:disc_idx[n]]
        elif n == len(disc_idx):
            signals = trend_components[:, disc_idx[n - 1]:]
            stim_off_idx = stim_off[disc_idx[n - 1]:]
        else:
            signals = trend_components[:, disc_idx[n - 1]:disc_idx[n]]
            stim_off_idx = stim_off[disc_idx[n - 1]:disc_idx[n]]

        # Only fit to samples when stim is off
        targets = signals[:, stim_off_idx].T
        dims = [len(stim_off_idx), 1]

        # Fit quadratic to first trend when decay is strongest, linear to rest
        if n == 0 or (all_quad and n < len(disc_idx)):
            X = np.hstack([np.ones(dims),
                           np.arange(dims[0]).reshape(dims),
                           np.power(np.arange(dims[0]), 2).reshape(dims)])
        elif all_quad:
            X = np.hstack([np.ones(dims),
                           np.arange(dims[0]).reshape(dims),
                           np.log(np.arange(dims[0]) + 100).reshape(dims)])
        else:
            X = np.hstack([np.ones(dims), np.arange(dims[0]).reshape(dims)])

        # Make predictions for whole segment
        betas = np.linalg.inv(X[stim_off_idx, :].T.dot(X[stim_off_idx, :])).dot(
            X[stim_off_idx, :].T.dot(targets))
        predictions = X.dot(betas).T

        # Record Trend
        if n == 0:
            bleach_trend[:, :disc_idx[n]] = predictions
        elif n == len(disc_idx):
            bleach_trend[:, disc_idx[n - 1]:] = predictions
        else:
            bleach_trend[:, disc_idx[n - 1]:disc_idx[n]] = predictions

    # flag points for removal
    del_idx = np.empty(0)
    for disc in disc_idx:
        del_idx = np.append(del_idx, np.arange(disc - 3, disc + 4))

    return bleach_trend, del_idx.astype(int)


def thresh_mad(mov, x=3, axis=-1):
    mov = mov.copy()
    med_image = np.median(mov, axis=-1)
    mad_image = np.median(np.abs(mov - med_image[:, :, np.newaxis]), axis=-1)
    mov[mov < (med_image + (x * mad_image))[:, :, np.newaxis]] = 0
    return mov

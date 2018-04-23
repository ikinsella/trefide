import numpy as np
import scipy as sp

import trefide.extras.tools as tools_

# cython


def kurto_one(x):
    """
    Computer kurtosis of rows of x with 2D (d x T)
    Outputs: idx of components with kurtosis > population mean + std
    """
    # Compute kurtosis
    kurt = sp.stats.kurtosis(x, axis=1, fisher=True, bias=True)
    # keep component wrt mean + std
    kurt_th = kurt.mean() + kurt.std()
    keep = np.where(kurt > kurt_th)[0].tolist()
    return keep

def stimulus_segmentation(T,
                          stim_knots=None,
                          stim_delta=10):
    """
    Returns boolean areas to ignore in frame
    """
    if stim_knots is None:
        return np.zeros(T).astype('bool')

    stim_knots = np.where(np.abs(np.diff(stim_knots))>1)[0]

    ignore_frames =np.zeros(T)
    for knot in stim_knots:
        arg_np = (np.ones(stim_delta).cumsum()-stim_delta//2+knot).astype('int')
        ignore_frames[arg_np]=1
    ignore_frames=ignore_frames.astype('bool')
    return ignore_frames


def mean_confidence_interval(data,
                            confidence=0.90,
                            one_sided=False):
    """
    Compute mean confidence interval (CI)
    for a normally distributed population
    _____________

    Parameters:
    ___________
    data:       np.array  (L,)
                input vector from which to calculate mean CI
                assumes gaussian-like distribution
    confidence: float
                confidence level for test statistic
    one_sided:  boolean
                enforce a one-sided test

    Outputs:
    _______
    th:         float
                threshold for mean value at CI
    """
    if one_sided:
        confidence = 1 - 2*(confidence-1)
    _, th = sp.stats.norm.interval(confidence,loc =np.mean(data),scale=data.std())
    return th


def choose_rank(Vt,
                maxlag=10,
                iterate=False,
                confidence=0.90,
                corr=True,
                kurto=False,
                mean_th=None,
                mean_th_factor=1.,
                min_rank=0,
               enforce_both=False):
    """
    Select rank vectors in Vt which pass test statistic(s) enabled
    (e.g. axcov and/or kurtosis)

    __________
    Parameters:

    Vt:         np.array (k x T)
                array of k temporal components lasting T samples
    maxlag:     int
                max correlation lag for correlation null hypothesis in samples
                (e.g. indicator decay in samples)
    iterate:    boolean
                flag to include correlated components iteratively
    confidence: float
                confidence interval (CI) for correlation null hypothesis
    corr:       boolean
                flag to include components which pass correlation null hypothesis
    kurto:      boolean
                flag to include components which pass kurtosis null hypothesis
    mean_th:    float
                threshold employed to reject components according to
                correlation null hypothesis
    mean_th_factor: float
                factor to scale mean_th
                typically set to 2 if greedy=True and mean_th=None or
                if mean_th has not been scaled yet.
    min_rank:   int
                minimum number of components to include in output
                even if no components of Vt pass any test

    Outputs:
    -------

    vtid:       np.array (3,d)
                indicator 3D matrix (corr-kurto-reject) which points which statistic
                a given component passed and thus it is included.
                can vary according to min_rank

    """
    if enforce_both:
        corr= True
        kurto = True
        
    n, L = Vt.shape
    vtid = np.zeros(shape=(3, n)) * np.nan

    # Null hypothesis: white noise ACF
    if corr is True:
        if mean_th is None:
            mean_th = wnoise_acov_CI(L,confidence=confidence,maxlag=maxlag)
        mean_th*= mean_th_factor
        keep1 = vector_acov(Vt,
                            mean_th = mean_th,
                            maxlag=maxlag,
                            iterate=iterate,
                            min_rank=min_rank)
    else:
        keep1 = []
    if kurto is True:
        keep2 = kurto_one(Vt)
    else:
        keep2 = []

    keep = list(set(keep1 + keep2))
    loose = np.setdiff1d(np.arange(n),keep)
    loose = list(loose)
    
    if enforce_both:
        keep1 = np.intersect1d(keep1,keep2)
        keep2 = keep1
        print(len(keep1))
    

    vtid[0, keep1] = 1  # components stored due to cov
    vtid[1, keep2] = 1  # components stored due to kurto
    vtid[2, loose] = 1  # extra components ignored
    # print('rank cov {} and rank kurto {}'.format(len(keep1),len(keep)-len(keep1)))
    return vtid


def wnoise_acov_CI(L,
                    confidence=0.90,
                    maxlag=10,
                    n=3000,
                    plot_en=False):
    """
    Generate n AWGN vectors lasting L samples.
    Calculate the mean of the ACF of each vector for 0:maxlag
    Return the mean threshold with specified confidence.

    Parameters:
    ----------

    L:          int
                length of vector
    confidence: float
                confidence level for test statistic
    maxlag:     int
                max correlation lag for correlation null hypothesis
                in samples (e.g. indicator decay in samples)
    n:          int
                number of standard normal vectors to generate

    plot_en:    boolean
                plot histogram of pd
    Outputs:
    -------

    mean_th:    float
                value of mean of ACFs of each standard normal vector at CI.
    """
    # th1 = 0
    #print 'confidence is {}'.format(confidence)
    covs_ht = np.zeros(shape=(n,))
    for sample in np.arange(n):
        ht_data = np.random.randn(L)
        covdata = tools_.axcov(ht_data, maxlag)[maxlag:]/ht_data.var()
        covs_ht[sample] = covdata.mean()
        #covs_ht[sample] = np.abs(covdata[1:]).mean()
    #hist, _,_=plt.hist(covs_ht)
    #plt.show()
    mean_th = mean_confidence_interval(covs_ht, confidence)
    #print('th is {}'.format(mean_th))
    return mean_th


def vector_acov(Vt,
                mean_th=0.10,
                maxlag=10,
                iterate=False,
                extra=1,
                min_rank=0,
                verbose=False):
    """
    Calculate auto covariance of row vectors in Vt
    and output indices of vectors which pass correlation null hypothesis.

    Parameters:
    ----------
    Vt:         np.array(k x T)
                row array of compoenents on which to test correlation null hypothesis
    mean_th:    float
                threshold employed to reject components according to correlation null hypothesis
    maxlag:     int
                determined lag until which to average ACF of row-vectors for null hypothesis
    iterate:    boolean
                flag to include components which pass null hypothesis iteratively
                (i.e. if the next row fails, no additional components are added)
    extra:      int
                number of components to add as extra to components which pass null hypothesis
                components are added in ascending order corresponding to order in mean_th
    min_rank:   int
                minimum number of components that should be included
                add additional components given components that (not) passed null hypothesis
    verbose:    boolean
                flag to enable verbose

    Outputs:
    -------
    keep:       list
                includes indices of components which passed the null hypothesis
                and/or additional components added given parameters
    """

    keep = []
    num_components = Vt.shape[0]
    print('mean_th is %s'%mean_th) if verbose else 0
    for vector in range(0, num_components):
        # standarize and normalize
        vi = Vt[vector, :]
        vi =(vi - vi.mean())/vi.std()
        print('vi mean = %.3f var = %.3f'%(vi.mean(),vi.var())) if verbose else 0
        vi_cov = tools_.axcov(vi, maxlag)[maxlag:]/vi.var()
        print(vi_cov.mean()) if verbose else 0
        if vi_cov.mean() < mean_th:
            if iterate is True:
                break
        else:
            keep.append(vector)
    # Store extra components
    if vector < num_components and extra != 1:
        extra = min(vector*extra,Vt.shape[0])
        for addv in range(1, extra-vector+ 1):
            keep.append(vector + addv)
    # Forcing one components
    if not keep and min_rank>0:
        # vector is empty for once min
        keep.append(0)
        print('Forcing one component') if verbose else 0
    return keep


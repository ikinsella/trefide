import cvxpy as cp
import numpy as np

from caiman.source_extraction.cnmf.pre_processing import get_noise_fft


def denoise(Y,
            noise_estimator='boot',
            noise_method='median',
            n_bs=1000,
            t_bs=25):
    """Denoiser for pixel/temporal components pre/post CNMF"""
    _, T = Y.shape

    # Default Difference Operator
    diff = (np.diag(2 * np.ones(T), 0) + np.diag(-1 * np.ones(T - 1), 1) +
            np.diag(-1 * np.ones(T - 1), -1))[1:T - 1]

    # Estimate Noise Standard Deviations
    noise_estimators = {
        'fft': lambda Y: get_noise_fft(Y,
                                       noise_method=noise_method)[0],
        'boot': lambda Y: get_noise_boot(Y,
                                         noise_method=noise_method,
                                         n_bs=n_bs,
                                         t_bs=t_bs)
    }
    stdvs = noise_estimators[noise_estimator](Y)

    # Denoise
    Y_hat = [c_l1tf(Y[idx, :], diff, stdv) for idx, stdv in enumerate(stdvs)]
    return np.array(Y_hat)


def c_l1tf(y, diff, sigma):
    """Use cvx to solve the constrained l1 trend filtering problem"""
    y_hat = cp.Variable(len(y))
    cp.Problem(
        cp.Minimize(cp.norm(cp.matmul(diff, y_hat), 1)),
        [cp.norm(y - y_hat, 2) <= sigma * np.sqrt(len(y))]).solve()
    return y_hat.value


def get_noise_boot(Y, n_bs=1000, t_bs=25, noise_method='mean'):
    """Estimate noise stdvs from bootstrapped linear regression fits"""
    X = np.array([np.arange(t_bs), np.ones(t_bs)]).T
    Hat = np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)
    stdvs = [
        _locally_linear_bootstrap(Y[idx, :], Hat, n_bs, t_bs, noise_method)
        for idx in range(Y.shape[0])
    ]
    return np.sqrt(stdvs)


def _locally_linear_bootstrap(y, Hat, n, t, method):
    """ bootstrap local linear regression estiamtes of signal noise var"""
    methods = {
        'mean': np.mean,
        'median': np.median,
        'logmexp': lambda x: np.exp(np.mean(np.log(x)))
    }
    mses = [
        np.mean(np.power(y[sdx:sdx + t] - np.dot(Hat, y[sdx:sdx + t]), 2))
        for sdx in np.random.randint(0, len(y) - t + 1, size=n)
    ]
    return methods[method](mses)

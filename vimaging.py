import cvxpy as cp
import numpy as np

from caiman.source_extraction.cnmf.pre_processing import get_noise_fft


def trend_filter_denoise(Y):
    """Denoiser for pixel/temporal components pre/post CNMF"""
    N, T = Y.shape

    # Default Difference Operator
    D = (np.diag(2 * np.ones(T), 0) + np.diag(-1 * np.ones(T - 1), 1)
         + np.diag(-1 * np.ones(T - 1), -1))[1:T - 1]

    # Estimate Noise Standard Deviations
    stdvs = get_noise_fft(Y, noise_method='mean')[0]

    # Denoise Filters
    Y_hat = [_solve_l1tf(Y[idx, :], D, stdv) for idx, stdv in enumerate(stdvs)]
    return(np.array(Y_hat))

def _solve_l1tf(y, D, sigma):
    """Uses cvx to solve the constrained l1tf problem"""
    T = len(y)
    y_hat = cp.Variable(T)
    _ = cp.Problem(cp.Minimize(cp.norm(cp.matmul(D, y_hat), 1)),
                   [cp.norm(y - y_hat, 2) <= sigma * np.sqrt(T)]).solve()
    return(y_hat.value)


# Implement Bootstrap methods below
# def _estimate_noise_stdv(Y):
#     """Estimates the standard deviation of the noise from high frequency
#     components of the signal's power spectral density"""

#     sp.signal.welch

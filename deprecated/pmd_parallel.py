import math

import numpy as np
import scipy.io as io
import multiprocessing as mp

from trefide.temporal import TrendFilter
from numpy.linalg import svd
from trefide.utils.greedyPCA import choose_rank
from prox_tv import tv1_2d


def eval_tv(image):
    """ Evaluates Anisotropic Total Variation of an image """
    return np.sum(np.abs(image[1:, :] - image[:-1, :])) +\
            np.sum(np.abs(image[:, 1:] - image[:, :-1]))


def update_temporal(R_k, u_k, trend_filter):
    """ Regress spatial against residual and denoise """
    v_k = np.dot(R_k.T, u_k)
    v_k = trend_filter.denoise(v_k)
    v_k /= np.linalg.norm(v_k)
    return v_k


def update_spatial(R_k, v_k, lambda_tv, dims):
    """ Regress temporal against redisual and denoise"""
    u_k = np.dot(R_k, v_k)
    u_k /= np.linalg.norm(u_k)
    u_k = tv1_2d(u_k.reshape(dims),
                 w=lambda_tv,
                 max_iters=1).reshape(np.prod(dims))
    u_k /= np.linalg.norm(u_k)
    return u_k


def pmd_tf_tv(Y,
              max_components=30,
              max_iters=50,
              lambda_tv=.0025,
              tol=5e-3):
    """ Run penalize matrix decomposition on a block """

    # Initialize Internal Vars
    d1, d2, T = Y.shape
    spatial_cutoff = (d1*d2 / ((d1*(d2-1) + d2*(d1-1))))*1.05
    R_k = Y.reshape(d1*d2, T)

    # Initialize Outputs
    U = np.empty((max_components, d1*d2))
    S = np.empty(max_components)
    V = np.empty((max_components, T))

    # Outer Loop: Extract Components
    num_components = 0
    for k in range(max_components):

        # Intialize Components & Filter
        trend_filter = TrendFilter(T)
        u_k = np.ones(d1*d2) / math.sqrt(d1*d2)
        v_k = update_temporal(R_k, u_k, trend_filter)
        s_k = u_k.T.dot(R_k.dot(v_k))
        v__ = v_k
        u__ = u_k
        s__ = s_k

        # Inner Loop: Refine Components
        for iter_ in range(max_iters):
            u_k = update_spatial(R_k, v_k, lambda_tv, (d1, d2))
            v_k = update_temporal(R_k, u_k, trend_filter)
            s_k = u_k.T.dot(R_k).dot(v_k)

            # Check For Convergence
            if np.max([np.abs((s_k - s__) / s__),
                       np.linalg.norm(v_k - v__),
                       np.linalg.norm(u_k - u__)]) < tol:
                break

            # Check If We're Fitting Noise
            if iter_ == 9:
                spatial_quality = np.sum(np.abs(u_k)) / eval_tv(u_k.reshape((d1, d2)))
                if spatial_quality < spatial_cutoff:
                    break

            # Update Components History
            v__ = v_k
            u__ = u_k
            s__ = s_k

        # Check If Spatial Components Is Noise
        spatial_quality = np.sum(np.abs(u_k)) / eval_tv(u_k.reshape((d1, d2)))
        if spatial_quality < spatial_cutoff:
            break
        else:
            # Debias Temporal & Store Components
            U[k, :] = u_k
            V[k, :] = R_k.T.dot(U[k, :])
            V[k, :] /= np.linalg.norm(V[k, :])
            S[k] = np.dot(np.dot(R_k, V[k, :]).T, U[k, :])
            R_k -= S[k] * U[k, :][:, None].dot(V[k, :][None, :])
            print(num_components)
            num_components += 1

    # Terminated, Return Denoised Components
    return U[:num_components, :], S[:num_components], V[:num_components, :]


if __name__ == "__main__":


    # Set block params
    height = 40
    width = 200

    # Make Internal Function
    indices = np.transpose([np.tile(range(int(d1/height)), int(d2/width)),
                            np.repeat(range(int(d2/width)), int(d1/height))])
    for idx, jdx in indices:
        p = mp.Process(target=pmd_tf_tv,
                       args=(Y_detr[idx * height:(idx+1) * height,
                                    jdx * width:(jdx+1) * width, :],))
        p.start()

import numpy as np
from trefide.pmd import batch_decompose
from trefide.utils import psd_noise_estimate

def test_psd_noise_estimate():
    n, t = 100, 5
    Y = np.random.randn(n, t)

    res = np.asarray(psd_noise_estimate(Y))

    print(res)

def main():
    test_psd_noise_estimate()

    X = np.load("/home/sy2685/trefide/data/demoMovie.npy")
    X = np.copy(X[:, :, :127])
    X = np.tile(X, (10, 1))

    d1, d2, T = X.shape
    print("movie shape: {}".format(X.shape))

    K = 3 # 50
    maxiter = 50
    consec_failures = 3
    tol = 5e-3
    bheight = 40
    bwidth = 40
    spatial_cutoff = (bheight*bwidth / ((bheight*(bwidth-1) + bwidth*(bheight-1))))
    w = .0025

    U, V, K, indices = batch_decompose(d1, d2, T, X, bheight, bwidth, w,
                                       spatial_cutoff, K, consec_failures,
                                       maxiter, maxiter, tol)
if __name__ == "__main__":
    main()

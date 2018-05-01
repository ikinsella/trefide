import numpy as np
from trefide.pmd import batch_decompose

if __name__ == "__main__":
    X = np.load("/home/ian/devel/trefide/data/prepared_sampleMovie.npy")
    X = np.tile(X, (10, 1))
    d1, d2, T = X.shape

    K = 50
    maxiter = 50
    consec_failures = 3
    tol = 5e-3
    bheight = 40
    bwidth = 40
    spatial_cutoff = (bheight*bwidth / ((bheight*(bwidth-1) + bwidth*(bheight-1))))
    w = .0025
    
    U, V, K, indices = batch_decompose(d1, d2, T, X, bheight, bwidth, w, spatial_cutoff, K, consec_failures, maxiter, tol)

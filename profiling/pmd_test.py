import numpy as np
from trefide.pmd import blockwise_pmd, parallel_pmd

if __name__ == "__main__":
    X = np.load("/home/ian/devel/trefide/data/prepared_sampleMovie.npy")
    d1, d2, T = X.shape

    K = 20
    maxiter = 50
    tol = 5e-3
    bheight = 40
    bwidth = 200
    spatial_cutoff = (bheight*bwidth / ((bheight*(bwidth-1) + bwidth*(bheight-1))))
    w = .0025
    
    #U, V, K, indices = blockwise_pmd(d1, d2, T, X, bheight, bwidth, w, spatial_cutoff, K, maxiter, tol)
    U, V, K, indices = parallel_pmd(d1, d2, T, X, bheight, bwidth, w, spatial_cutoff, K, maxiter, tol)

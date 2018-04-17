import scipy.io as io
import numpy as np
from trefide.pmd import factor_patch
from trefide.utils.noise import estimate_noise

if __name__ == "__main__":
    # Load & Process Dataset
    Y_detr = io.loadmat("/home/ian/devel/trefide/data/detrended_sampleMovie.mat")['Y_detr']
    d1, d2, T = Y_detr.shape
    Y_detr = (Y_detr.reshape(d1*d2, T) / estimate_noise(Y_detr.reshape(d1*d2, T), summarize='mean')[:, None]).reshape(d1, d2, T)

    # Set block params
    height = 20
    width = 100

    # Initialize Internal Vars
    K = 20
    U = np.zeros((K, height, width))
    V = np.zeros((K, T))
    maxiter = 50
    tol = 5e-3
    spatial_cutoff = (height*width / ((height*(width-1) + width*(height-1))))*1.05
    w = .0025

    for idx in range(int(d1/height)):
        for jdx in range(int(d2/width)):

            # Process Block
            print((idx, jdx))
            factor_patch(height,
                         width,
                         T,
                         Y_detr[idx * height:(idx+1) * height,
                                jdx * width:(jdx+1) * width, :].copy(),
                         U,
                         V,
                         w,
                         spatial_cutoff,
                         K,
                         maxiter,
                         tol)

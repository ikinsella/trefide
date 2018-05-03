import numpy as np
from trefide.pmd import batch_decompose

from trefide.decimation import decimated_decompose, decimated_batch_decompose, downsample_video, downsample_image, downsample_signal


def downsample_movie(fov_height, fov_width, d_sub, num_frames, t_sub, mov):
    Y_ds = np.asarray([downsample_image(fov_height, fov_width, d_sub, np.asfortranarray(frame)) 
                       for frame in mov.transpose(2,0,1)])
    Y_ds = np.asarray([downsample_signal(num_frames, t_sub, np.ascontiguousarray(signal))
                       for signal in np.reshape(Y_ds.transpose(1,2,0),
                                                (int(fov_height*fov_width/(d_sub**2)), num_frames))])
    return np.reshape(Y_ds, (int(fov_height/d_sub), int(fov_width/d_sub), int(num_frames/t_sub)))


if __name__ == "__main__":
    X = np.load("/home/ian/devel/trefide/data/prepared_sampleMovie.npy")
    d1, d2, T = X.shape
    d_sub = 2
    t_sub = 2
    X_ds = np.ascontiguousarray(downsample_movie(d1, d2, d_sub, T, t_sub, np.asfortranarray(X)))
    X = np.tile(X, (10, 1))
    X_ds = np.tile(X_ds, (10, 1))
    d1, d2, T = X.shape

    K = 50
    max_iters = 10
    max_iters_ds = 40
    consec_failures = 3
    tol = 5e-3
    bheight = 40
    bwidth = 40
    w = .0025
    
    U, V, K, indices = decimated_batch_decompose(d1, d2, d_sub, T, t_sub, X, X_ds, bheight, bwidth, w,  K, consec_failures, max_iters, max_iters_ds, tol)

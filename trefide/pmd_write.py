import argparse

import numpy as np
import scipy.io as io

from trefide.utils.noise import estimate_noise

if __name__ == "__main__":
    
    # Parse Command Line Args
    parser = argparse.ArgumentParser(description='filename, blockheight, blockwidth')
    parser.add_argument('-f', dest='filename', type=str)
    parser.add_argument('-bh', dest='height', type=int, default=20)
    parser.add_argument('-bw', dest='width', type=int, default=100)
    args = parser.parse_args()
    print(args.width)

    # Load Data
    Y_detr = io.loadmat(os.path.join("/home/ian/devel/trefide/data",
                        args.filename))['Y_detr']
    d1, d2, T = Y_detr.shape

    # Preprocess
    #sigma = estimate_noise(Y_detr.reshape(d1*d2, T),
    #                       summarize='mean').reshape(d1, d2)[:, :, None]
    #Y_detr /= sigma

    # Save Each Block

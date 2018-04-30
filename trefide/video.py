import glob
import os
import subprocess
import shutil
import cv2

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.axes_grid1 import make_axes_locatable


# --------------------------------------------------------------------------- #
# ------------------- Video Writing/Watching Utilities ---------------------- #
# --------------------------------------------------------------------------- #


def write_mpl(mov_list, 
              filename, 
              fr=30, 
              horizontal=True, 
              titles=None):
    """ Write Movies Using Matplotlib & ffmpeg """
    
    # Declare & Assign Local Variables
    n_mov = len(mov_list)
    T = mov_list[0].shape[2]
    if titles is None:
        titles = ['']*n_mov
    delete_tmp = False
    
    #Compute scales
    mins = np.empty(n_mov)
    maxs = np.empty(n_mov)
    for mdx, mov in enumerate(mov_list):
        mins[mdx] = np.min(mov)
        maxs[mdx] = np.max(mov)

    # Create tmp directory as workspace
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
        delete_tmp = True

    # Plot & Save Frames
    for t in range(T):
        
        # Choose Layout
        if horizontal:
            fig, ax = plt.subplots(1, n_mov, figsize=(8,8))
        else:
            fig, ax = plt.subplots(n_mov, 1, figsize=(8,8))
        
        # Display Current Frame From Each Mov
        for mdx, mov in enumerate(mov_list):
            
            divider = make_axes_locatable(ax[mdx])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax[mdx].imshow(mov[:,:,t],
                                cmap=cm.Greys_r,
                                vmin=mins[mdx],
                                vmax=maxs[mdx])
            plt.colorbar(im, cax, orientation='vertical')
            ax[mdx].set_title(titles[mdx] + " Frame:{}".format(t))
        
        # Save Figure As PNG
        #plt.tight_layout()
        plt.savefig(os.path.join("tmp", filename + "%04d.png" % t))
        
        # Close Figure
        plt.close('all')
        
    # Call FFMPEG to compile PNGs
    os.chdir("tmp")
    subprocess.call(['ffmpeg',
                     '-framerate', str(fr),
                     '-i', filename + '%04d.png',
                     '-r', str(fr), 
                     '-pix_fmt', 'yuv420p',
                     filename + '.mp4'])
    if delete_tmp:
        shutil.copy2(filename + ".mp4", "..")
        os.chdir("..")
        shutil.rmtree('tmp')
    else:
        shutil.copy2(filename + ".mp4", "..")
        filelist = glob.glob(os.path.join("*.png"))
        for f in filelist:
            os.remove(f)
        os.remove(filename + ".mp4")
        os.chdir("../")


def play_cv2(movie, 
             gain=3, 
             fr=60, 
             offset=0, 
             magnification=1, 
             repeat=False):
    """ Render Video With OpenCV3 Library's Imshow"""
    T = movie.shape[2]
    maxmov = np.max(movie)
    looping=True
    terminated=False
    while looping:
        for t in range(T):
            if magnification != 1:
                frame = cv2.resize(movie[:,:,t],
                                   None,
                                   fx=magnification,
                                   fy=magnification,
                                   interpolation=cv2.INTER_LINEAR)
            else:
                frame = movie[:,:,t]
            cv2.imshow('frame', (frame - offset) / maxmov*gain)
            if cv2.waitKey(int(1. / fr * 1000)) & 0xFF == ord('q'):
                looping = False
                terminated = True
                break
        if terminated:
            break
        looping=repeat

    cv2.waitKey(100)
    cv2.destroyAllWindows()
    for i in range(10):
        cv2.waitKey(100)

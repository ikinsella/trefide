import numpy as np

def weighted_component_reformat(U, V, K, indices, W):
    """ Reconstruct A Denoised Movie """

    # Get Block Size Info From Spatial
    num_blocks = U.shape[0]
    bheight = U.shape[1]
    bwidth = U.shape[2]
    t = V.shape[2]

    # Get Mvie Size Infro From Indices
    nbi = len(np.unique(indices[:,0]))
    idx_offset = np.min(indices[:,0])
    nbj = len(np.unique(indices[:,1]))
    jdx_offset = np.min(indices[:,1])
    d1 = nbi * bheight
    d2 = nbj * bwidth

    # compute total ranks
    total_ranks = 0
    for bdx in range(nbi*nbj):
        total_ranks += K[bdx]
    total_ranks = int(total_ranks)

    # Allocate Space For reconstructed Movies
    U_full = np.zeros((d1, d2, total_ranks), dtype=np.float64)
    V_full = np.zeros((total_ranks, t), dtype=np.float64)

    # Loop Over Blocks
    k = 0
    for bdx in range(nbi*nbj):
        idx = int((indices[bdx,0] - idx_offset) * bheight)
        jdx = int((indices[bdx,1] - jdx_offset) * bwidth)
        for k_bdx in range(K[bdx]):
            U_full[idx:idx+bheight, jdx:jdx+bwidth, k] += U[bdx, :, :, k_bdx] * np.asarray(W)
            V_full[k] = V[bdx, k_bdx, :]
            k += 1
    return U_full, V_full


def overlapping_component_reformat(d1,
                                   d2,
                                   t,
                                   bheight,
                                   bwidth,
                                   U, V, K, I, W):
    """ 4x batch denoiser """
   
    # Assert Even Blockdims
    assert bheight % 2 == 0 , "Block height must be an even integer."
    assert bwidth % 2 == 0 , "Block width must be an even integer."
    
    # Assert Even Blockdims
    assert d1 % bheight == 0 , "Input FOV height must be an evenly divisible by block height."
    assert d2 % bwidth == 0 , "Input FOV width must be evenly divisible by block width."
    
    # Declare internal vars
    hbheight = int(bheight/2)
    hbwidth = int(bwidth/2)
    nbrow = int(d1/bheight)
    nbcol = int(d2/bwidth)

    # Count All ranks & Allocate Space
    total_rank = int(np.sum(K['no_skew']['full']) + 
                     np.sum(K['vert_skew']['full']) + 
                     np.sum(K['vert_skew']['half']) + 
                     np.sum(K['horz_skew']['full']) + 
                     np.sum(K['horz_skew']['half']) + 
                     np.sum(K['diag_skew']['full']) + 
                     np.sum(K['diag_skew']['thalf']) + 
                     np.sum(K['diag_skew']['whalf']) + 
                     np.sum(K['diag_skew']['quarter']))

    # Allocate Space For reconstructed Movies
    component_count = 0
    U_full = np.zeros((d1,d2,total_rank), dtype=np.float64)
    V_full = np.zeros((total_rank, t), dtype=np.float64)

    # ---------------- Handle Blocks Overlays One At A Time --------------#

    # ----------- Original Overlay --------------
    # Only Need To Process Full-Size Blocks
    K_ = int(np.sum(K['no_skew']['full']))
    U_, V_ = weighted_component_reformat(U['no_skew']['full'],
                                         V['no_skew']['full'], 
                                         K['no_skew']['full'], 
                                         I['no_skew']['full'], 
                                         W)
    U_full[:,:,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_

    # ---------- Vertical Skew --------------
    # Full Blocks
    K_ = int(np.sum(K['vert_skew']['full']))
    U_, V_ = weighted_component_reformat(U['vert_skew']['full'],
                                         V['vert_skew']['full'],
                                         K['vert_skew']['full'], 
                                         I['vert_skew']['full'], 
                                         W)
    U_full[hbheight:d1-hbheight,:,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    # wide half blocks
    K_ = int(np.sum(K['vert_skew']['half'][::2]))
    U_, V_ = weighted_component_reformat(U['vert_skew']['half'][::2],
                                         V['vert_skew']['half'][::2],
                                         K['vert_skew']['half'][::2],
                                         I['vert_skew']['half'][::2],
                                         W[hbheight:, :])
    U_full[:hbheight,:,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    K_ = int(np.sum(K['vert_skew']['half'][1::2]))
    U_, V_ = weighted_component_reformat(U['vert_skew']['half'][1::2],
                                         V['vert_skew']['half'][1::2],
                                         K['vert_skew']['half'][1::2],
                                         I['vert_skew']['half'][1::2],
                                         W[:hbheight, :])
    U_full[d1-hbheight:,:,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 
    
    # --------------Horizontal Skew--------------
    # Full Blocks
    K_ = int(np.sum(K['horz_skew']['full']))
    U_, V_ = weighted_component_reformat(U['horz_skew']['full'],
                                         V['horz_skew']['full'],
                                         K['horz_skew']['full'],
                                         I['horz_skew']['full'], 
                                         W)
    U_full[:,hbwidth:d2-hbwidth,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    # tall half blocks
    K_ = int(np.sum(K['horz_skew']['half'][:nbrow]))
    U_, V_ = weighted_component_reformat(U['horz_skew']['half'][:nbrow],
                                         V['horz_skew']['half'][:nbrow], 
                                         K['horz_skew']['half'][:nbrow], 
                                         I['horz_skew']['half'][:nbrow],  
                                         W[:, hbwidth:])
    U_full[:,:hbwidth,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    K_ = int(np.sum(K['horz_skew']['half'][nbrow:]))
    U_, V_ = weighted_component_reformat(U['horz_skew']['half'][nbrow:],
                                         V['horz_skew']['half'][nbrow:],
                                         K['horz_skew']['half'][nbrow:],
                                         I['horz_skew']['half'][nbrow:],
                                         W[:, :hbwidth])
    U_full[:,d2-hbwidth:,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    # -------------Diagonal Skew--------------
    # Full Blocks
    K_ = int(np.sum(K['diag_skew']['full']))
    U_, V_ = weighted_component_reformat(U['diag_skew']['full'],
                                         V['diag_skew']['full'],
                                         K['diag_skew']['full'],
                                         I['diag_skew']['full'],
                                         W)
    U_full[hbheight:d1-hbheight, hbwidth:d2-hbwidth,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    # tall half blocks
    K_ = int(np.sum(K['diag_skew']['thalf'][:nbrow-1]))
    U_, V_ = weighted_component_reformat(U['diag_skew']['thalf'][:nbrow-1],
                                         V['diag_skew']['thalf'][:nbrow-1], 
                                         K['diag_skew']['thalf'][:nbrow-1], 
                                         I['diag_skew']['thalf'][:nbrow-1],  
                                         W[:, hbwidth:])
    U_full[hbheight:d1-hbheight,:hbwidth,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    K_ = int(np.sum(K['diag_skew']['thalf'][nbrow-1:]))
    U_, V_ = weighted_component_reformat(U['diag_skew']['thalf'][nbrow-1:], 
                                         V['diag_skew']['thalf'][nbrow-1:], 
                                         K['diag_skew']['thalf'][nbrow-1:], 
                                         I['diag_skew']['thalf'][nbrow-1:], 
                                         W[:, :hbwidth])
    U_full[hbheight:d1-hbheight,d2-hbwidth:,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    # wide half blocks
    K_ = int(np.sum(K['diag_skew']['whalf'][::2]))
    U_, V_ = weighted_component_reformat(U['diag_skew']['whalf'][::2], 
                                V['diag_skew']['whalf'][::2], 
                                K['diag_skew']['whalf'][::2], 
                                I['diag_skew']['whalf'][::2],  
                                W[hbheight:, :])
    U_full[:hbheight,hbwidth:d2-hbwidth,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    K_ = int(np.sum(K['diag_skew']['whalf'][1::2]))
    U_, V_ = weighted_component_reformat(U['diag_skew']['whalf'][1::2], 
                                         V['diag_skew']['whalf'][1::2], 
                                         K['diag_skew']['whalf'][1::2], 
                                         I['diag_skew']['whalf'][1::2], 
                                         W[:hbheight, :])
    U_full[d1-hbheight:,hbwidth:d2-hbwidth,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    # Corners
    K_ = int(np.sum(K['diag_skew']['quarter'][:1]))
    U_, V_ = weighted_component_reformat(U['diag_skew']['quarter'][:1],
                                         V['diag_skew']['quarter'][:1],
                                         K['diag_skew']['quarter'][:1],
                                         I['diag_skew']['quarter'][:1],
                                         W[hbheight:, hbwidth:])
    U_full[:hbheight,:hbwidth,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    K_ = int(np.sum(K['diag_skew']['quarter'][1:2]))
    U_, V_ = weighted_component_reformat(U['diag_skew']['quarter'][1:2],
                                         V['diag_skew']['quarter'][1:2],
                                         K['diag_skew']['quarter'][1:2],
                                         I['diag_skew']['quarter'][1:2],
                                         W[:hbheight, hbwidth:])
    U_full[d1-hbheight:,:hbwidth,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    K_ = int(np.sum(K['diag_skew']['quarter'][2:3]))
    U_, V_ = weighted_component_reformat(U['diag_skew']['quarter'][2:3], 
                                         V['diag_skew']['quarter'][2:3], 
                                         K['diag_skew']['quarter'][2:3], 
                                         I['diag_skew']['quarter'][2:3],  
                                         W[hbheight:, :hbwidth])
    U_full[:hbheight,d2-hbwidth:,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 

    K_ = int(np.sum(K['diag_skew']['quarter'][3:]))
    U_, V_ = weighted_component_reformat(U['diag_skew']['quarter'][3:], 
                                         V['diag_skew']['quarter'][3:], 
                                         K['diag_skew']['quarter'][3:], 
                                         I['diag_skew']['quarter'][3:],  
                                         W[:hbheight:, :hbwidth])
    U_full[d1-hbheight:,d2-hbwidth:,component_count:component_count+K_] += U_
    V_full[component_count:component_count+K_,:] = V_
    component_count += K_ 
    return U_full, V_full

import numpy as np
import pandas as pd
import hcp_utils as hcp
from math import inf
import scipy.io
import os

from .preparedata import get_prob_matrix
from .sim_annealing import bigphi_calc

# Starting from a probability voxel matrix, we find the maximum correlation for each voxel (df_max)

# ---------------------------------------------------------------------------------
# Big Phi functions
# ---------------------------------------------------------------------------------
def Big_Phi_Sasai(Pi_max):
    Pmin_max = Pi_max[-1]
    Pi_ex = Pi_max[0:-1]
    Bphi = np.sum(np.log10(Pi_max)) + np.log10(Pmin_max) + np.sum(np.log10(1 + Pi_ex))
    return Bphi

def Big_Phi_new(Pi_max, debug = False):
    Pmin_max = Pi_max[-1]
    Pi_ex = Pi_max[0:-1]
    
    Bphi = np.sum(np.log10(1 + Pi_max)) - len(Pi_max) * np.log10(2) + np.log10(Pmin_max) + np.sum(np.log10(1 + Pi_ex))
    if debug is True:
        term1 = np.sum(np.log10(1 + Pi_max))
        term2 =  np.log10(2)*len(Pi_max)
        term3 = np.log10(Pmin_max) + np.sum(np.log10(1 + Pi_ex))
        term4 = np.sum(np.log10(Pi_max))
        return Bphi, term1, term2, term3, term4
    else: 
        return Bphi

# ---------------------------------------------------------------------------------
# main evaluations functions
# ---------------------------------------------------------------------------------

def major_complex(P_cor, big_phi_function = Big_Phi_new):
    df_max = prepare_max_probabilities(P_cor)
    return big_phi_faster(df_max, big_phi_function)

def subsystem(P_sub, big_phi_function = Big_Phi_new):
    df_max = prepare_max_probabilities(P_sub)
    return big_phi_function(df_max[1].values)

# ---------------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------------
def prepare_max_probabilities(P_cor, cut_off = 0.0):
    """
    Args:
        P_cor (np.ndarray): A probability matrix from the vertice covariance matrix
        cut_off (float): The percentile (0.001 corresponds to top 1/1000th) in identifying the "cut" (this controls for extreme outlier correlations)

    Returns:
        df_max (pandas database): sorted list of maximum correlation (column name 0) with the respective pair of voxels saved as max_idx and min_idx 
    """
    #Issue: Code doesn't work with cut_off because for different voxels the cut off is different. 
    # So the maxima don't pair up and the number of rows doesn't necessarily correspond to the #voxels
    df = pd.DataFrame(P_cor)
    #df = df.mask(df > (1-cut_off)*df.max())
    df_max = pd.concat([df.idxmax(),df.max()], axis = 1)
    df_max.reset_index(level = 0, inplace = True)
    df_max['max_idx'] = df_max[['index', 0]].max(axis = 1)
    df_max['min_idx'] = df_max[['index', 0]].min(axis = 1)
    df_max = df_max.drop(columns = ['index', 0])
    #df_max = df_max.drop_duplicates()
    df_max = df_max.sort_values(by = [1], ascending = False)
    return df_max

# ---------------------------------------------------------------------------------
# Find main complex
# ---------------------------------------------------------------------------------
def big_phi_faster(df_max, big_phi_function = Big_Phi_new):
    Bphi_old = -inf
    for r in range(2,len(df_max)):
        #print(r, 'Phi: ', Bphi_old)

        Bphi = big_phi_function(df_max[1][0:r].values)

        #print(r, 'Phi: ', Bphi_old)
        if Bphi > Bphi_old: 
            Bphi_old = Bphi
            # Complex = df_max[['max_idx', 'min_idx']][0:r].values
            # Complex = np.unique(Complex)
            # print(len(Complex))
        else: 
            Complex = df_max[['max_idx', 'min_idx']][0:r].values
            Complex = np.unique(Complex)
            return Bphi_old, Complex
    
    # if Bphi kept growing
    Complex = df_max[['max_idx', 'min_idx']].values
    Complex = np.unique(Complex)
    return Bphi_old, Complex

# ---------------------------------------------------------------------------------
# Shun fast agglomeration_str(Rmat)
# ---------------------------------------------------------------------------------

def fast_agglomeration_str(P_cor):
    N = len(P_cor)
    ind_last = np.zeros(N, dtype=int)
    # sum total correlation
    sum_total_r = np.sum(P_cor,1)
    #print(np.max(sum_total_r))
    # get voxel with highest sum correlation
    maxind = np.argmax(sum_total_r)
    ind_last[0] = maxind

    ind_all = np.array([n for n in range(N)])
    # take row of P_cor of the voxel with highest sum correlation
    Rvec = P_cor[maxind,:]

    for ii in range(1,N):
        # exclude all previously used voxels
        ind_all[ind_last[0:ii]] = -1
        ID_list = np.where(ind_all>=0)[0]
        # find next max
        maxind = np.argmax(Rvec[ID_list])
        ind_last[ii] = ID_list[maxind]
        # add correlations of new node to summed correlations of previous row
        # this means that the next node chosen has the highest summed correlation with all prior nodes
        Rvec = Rvec + P_cor[ID_list[maxind]]

    return ind_last

# ---------------------------------------------------------------------------------
# ROI evaluation wb front back cerebellum
# ---------------------------------------------------------------------------------

def get_ROI_F_P():
    this_dir, _ = os.path.split(__file__)
    F_full = scipy.io.loadmat(os.path.join(this_dir, "parcellations/CorticalArealMask.mat"))['F_full']-1 #Convert to python 0 based indexing
    P_full = scipy.io.loadmat(os.path.join(this_dir, "parcellations/CorticalArealMask.mat"))['P_full']-1

    mat = scipy.io.loadmat(os.path.join(this_dir, "parcellations/resampled_cortex_ID_3_4mm.mat"))
    # intersecting vertices across left and right hemispheres, 0-based index?
    intersect_vert = (
        np.intersect1d(mat["resample_hemi"][0][0][0], mat["resample_hemi"][1][0][0]) - 1
        )   
    D = len(intersect_vert)

    # Need indices in intersect that correspond to Front
    ROI_F = np.nonzero(np.in1d(intersect_vert, F_full))[0]
    ROI_P = np.nonzero(np.in1d(intersect_vert, P_full))[0]

    ROI_F = np.append(ROI_F,ROI_F+D, 0)
    ROI_P = np.append(ROI_P,ROI_P+D, 0)

    return ROI_F, ROI_P

def Phi_ROI_wb_front_back_cerebellum(files_cor, files_cer, num_voxels = 2692, cut_off = 0.001, big_phi_function = []):

    ROI_F, ROI_P = get_ROI_F_P()
    data = []
    for c in range(len(files_cor)):
        print(c, ' out of ', len(files_cor))
        cor_mat = scipy.io.loadmat(files_cor[c])
        cer_mat = scipy.io.loadmat(files_cer[c])

        S_cor = cor_mat['signal']
        S_cer = cer_mat['signal']

        P_cor, _ = get_prob_matrix(S_cor)
        P_cer, _ = get_prob_matrix(S_cer)

        P_front = P_cor[np.ix_(ROI_F, ROI_F)] 
        P_post =  P_cor[np.ix_(ROI_P, ROI_P)] 
               
        # only 2692 best voxels following Shun's fast agglomeration
        OROI_cer = fast_agglomeration_str(P_cer)[0:num_voxels]
        OROI_F = fast_agglomeration_str(P_front)[0:num_voxels]
        OROI_P = fast_agglomeration_str(P_post)[0:num_voxels]

        P_cer_op = P_cer[np.ix_(OROI_cer, OROI_cer)]
        P_front_op = P_front[np.ix_(OROI_F, OROI_F)] 
        P_post_op =  P_post[np.ix_(OROI_P, OROI_P)] 

        if cut_off > 0:
            # all voxels in region
            wb, _ = bigphi_calc(P_cor, cut_off)

            ce_op, _ = bigphi_calc(P_cer_op, cut_off)
            fr_op, _ = bigphi_calc(P_front_op, cut_off)
            po_op, _ = bigphi_calc(P_post_op, cut_off)

        else:
            wb = subsystem(P_cor, big_phi_function=big_phi_function)

            ce_op = subsystem(P_cer_op, big_phi_function=big_phi_function)
            fr_op = subsystem(P_front_op, big_phi_function=big_phi_function)
            po_op = subsystem(P_post_op, big_phi_function=big_phi_function)

        data.append([wb, ce_op, fr_op, po_op])

    df_subjects = pd.DataFrame(data, columns = ['wb', 'cerr', 'front', 'back']) 

    return df_subjects
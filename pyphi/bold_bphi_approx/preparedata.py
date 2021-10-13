import numpy as np
import hcp_utils as hcp
import os
from sklearn.covariance import OAS
import scipy

def resample(Xn):
    #  HCP fMRI data are defined on a subset of the surface vertices (29696 out of 32492 for the left 
    # cortex and 29716 out of 32492 for the right cortex). Hence we have to construct an 
    # auxilliary array of size 32492 or 64984 with the fMRI data points inserted in appropriate places 
    # and a constant (zero by default) elsewhere. This is achieved by the cortex_data(arr, fill=0), 
    # left_cortex_data(arr, fill=0) and right_cortex_data(arr, fill=0)
    
    this_dir, this_filename = os.path.split(__file__)
    mat = scipy.io.loadmat(os.path.join(this_dir, "parcellations/resampled_cortex_ID_3_4mm.mat"))

    # intersecting vertices across left and right hemispheres, 0-based index?
    intersect_vert = (
        np.intersect1d(mat["resample_hemi"][0][0][0], mat["resample_hemi"][1][0][0]) - 1
    )
    
    first_iter = 1
    for i in range(0,Xn.shape[0]):
        if first_iter:
            Xall_left = np.atleast_2d(hcp.left_cortex_data(Xn[i,hcp.struct.cortex_left], fill=0))
            first_iter = 0
        else:
            Xall_left = np.concatenate((Xall_left, 
                                np.atleast_2d(hcp.left_cortex_data(Xn[i,hcp.struct.cortex_left], fill=0))), axis=0)

    print("left finished") 
    first_iter = 1
    for i in range(0,Xn.shape[0]):
        if first_iter:
            Xall_right = np.atleast_2d(hcp.right_cortex_data(Xn[i,hcp.struct.cortex_right], fill=0))
            first_iter = 0
        else:
            Xall_right = np.concatenate((Xall_right, 
                                np.atleast_2d(hcp.right_cortex_data(Xn[i,hcp.struct.cortex_right], fill=0))), axis=0)

    print("right finished")

    Xall = np.concatenate((Xall_left[:,intersect_vert],
                           Xall_right[:,intersect_vert]),axis=1)
                           

    return Xall


def get_prob_matrix(Xall):
    # from sklearn.covariance import OAS
    oas = OAS().fit(Xall)

    def correlation_from_covariance(covariance):
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation

    cor_mat = correlation_from_covariance(oas.covariance_)

    def P1(x):
        return 2*((1/4)+(1/(2*np.pi))*np.arcsin(x))

    def P0(x):
        return 2*((1/4)-(1/(2*np.pi))*np.arcsin(x))

    # diagonals are 1, so we zero them out here
    P1_vals = P1(cor_mat)
    np.fill_diagonal(P1_vals,0)

    P0_vals = P0(cor_mat)
    np.fill_diagonal(P0_vals,0)

    return np.maximum(P1_vals,P0_vals), cor_mat

def get_full_lr_texture(vscore_all):

    this_dir, this_filename = os.path.split(__file__)
    mat = scipy.io.loadmat(os.path.join(this_dir, "parcellations/resampled_cortex_ID_3_4mm.mat"))

    # intersecting vertices across left and right hemispheres, 0-based index?
    intersect_resample = (
        np.intersect1d(mat["resample_hemi"][0][0][0], mat["resample_hemi"][1][0][0]) - 1
    )

    lf = np.zeros(32492)
    rf = np.zeros(32492)

    lf[intersect_resample] = vscore_all[0 : len(intersect_resample)]
    rf[intersect_resample] = vscore_all[len(intersect_resample) :]

    return lf, rf



import numpy as np
from sklearn.linear_model import LogisticRegression
from ..utils import all_states

def binarize_signal(signal):
    # signal: time x voxels
    # binarize such that each voxel has equal number of zeros and ones over time
    t_steps = signal.shape[0]
    # divide along the median 
    threshold = np.floor_divide(t_steps,2)
    # sort each column (voxel) get index
    sort_ind = np.argsort(signal, axis = 0)
    # put 1s into empty array for half of the indices
    B_signal = np.zeros(signal.shape)  
    np.put_along_axis(B_signal, sort_ind[threshold:,:], 1, axis = 0)
    
    return B_signal

def list_observed_states(B_signal, M):
    # B_signal: time x voxels
    # Observed states of the mechanism units over time
    # Turns binarized signal (B_signal) into numerical state (100) is 1
    size_m = len(M)
    t_steps = B_signal.shape[0] #Time

    M_signal = B_signal[:,M]
    
    state_list = np.zeros(t_steps)
    for ss in range(size_m):
        state_list = state_list + M_signal[:,ss]*(2**ss)
    
    return state_list

def logit_cause_tpm(B_signal, P_cor, M, size_c):
    # based on Shun's logit_coefficient_cause function
    # B_signal: time x voxels
    # P_cor: voxels x voxels
    # Find candidate purview (most correlated)
    # returns state by node tpm (p(M = 1))
    rest_vec = [i for i in range(len(P_cor)) if i not in M]
    purview = []

    if size_c <= len(M):
        P_cor_m_p = P_cor[np.ix_(M, rest_vec)]
        # Purview is max of correlation with mechanism nodes
        for ii in range(size_c):
            max_ind = np.argmax(P_cor_m_p[ii])
            purview.append(rest_vec[max_ind])

            P_cor_m_p = np.delete(P_cor_m_p, max_ind, axis = 1)
            rest_vec = np.delete(rest_vec, max_ind)

    else:
        print('P > M not implemented yet')

    all_states = [a for a in pyphi.utils.all_states(size_c)]

    #logistic regression
    tpm_p_m = []
    for m_node in M:
        lr = LogisticRegression()
        lr.fit(B_signal[:,purview], B_signal[:,m_node])
        # get probabilities for m_node to be on for all possible states of purview
        tpm_p_m.append(lr.predict_proba(all_states)[:,1])

    return np.transpose(tpm_p_m)
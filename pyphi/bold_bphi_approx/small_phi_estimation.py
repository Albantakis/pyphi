import numpy as np


def binarize_signal(signal):
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

def list_observed_states(Bsignal, M):
    # Observed states of the mechanism units over time
    # Turns binarized signal (Bsignal) into numerical state (100) is 1
    size_m = len(M)
    M_signal = Bsignal[M]
    t_steps = Bsignal.shape[1] #Time
    state_list = np.zeros(t_steps)
    for ss in range(size_m):
        state_list = state_list + M_signal[ss,:]*(2**ss)
    
    return state_list


#def phi_ID_cause_wo_BGC_1node_MIP(size_m, size_cp, ):

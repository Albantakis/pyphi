import numpy as np
import pandas as pd
import scipy.io
import json
import multiprocessing
from sklearn.linear_model import LogisticRegression
from random import sample

from .preparedata import get_prob_matrix
from ..utils import all_states
from ..network import Network
from ..subsystem import Subsystem


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

    state_list = state_list.astype(int)
    
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

    list_all_states = [s for s in all_states(size_c)]

    #logistic regression
    tpm_p_m = []
    for m_node in M:
        lr = LogisticRegression()
        lr.fit(B_signal[:,purview], B_signal[:,m_node])
        # get probabilities for m_node to be on for all possible states of purview
        tpm_p_m.append(lr.predict_proba(list_all_states)[:,1])

    return np.transpose(tpm_p_m)

def small_phi_cause(tpm, states, state_counts = 1):
    # compute cause small phi of highest order mechanism
    net = Network(tpm)

    phi = []
    purview_size = []
    all_states_tuples = [s for s in all_states(tpm.shape[1])]
    for state in states:
        subsystem = Subsystem(net, all_states_tuples[state])
        cause = subsystem.mic(subsystem.node_indices)
        phi.append(cause.phi)
        purview_size.append(len(cause.purview))

    if state_counts != 1:
        phi_mean = sum(phi * state_counts / sum(state_counts))
        purview_size_mean = sum(purview_size * state_counts / sum(state_counts))
        return phi_mean, purview_size_mean

    else:
        return phi, purview_size


def small_phi_across_orders(B_signal, P_cor, num_mechs = 100, max_M_size = 8, save_flag = False, save_path = './', subject_id = ''):
    
    phis =[]
    purview_sizes = []

    for m_size in range(1,max_M_size+1):
        print('order: ', m_size)

        phi_m = []
        purview_size_m = []

        Ms = []
        for n in range(num_mechs):
            Ms.append(sample(range(len(P_cor)), m_size))
        
        for m in Ms:
            tpm_p_m = logit_cause_tpm(B_signal,P_cor, m, m_size)

            state_list = list_observed_states(B_signal, m)
            unique_states, counts = np.unique(state_list, return_counts=True)        
            # max state only
            ind_most_common = np.argmax(counts)
            most_common_state = unique_states[ind_most_common]
            # TODO: all states       
            av_phi, av_purview_size = small_phi_cause(tpm_p_m, [most_common_state])

            phi_m.extend(av_phi)
            purview_size_m.extend(av_purview_size)

        phis.append(phi_m)
        purview_sizes.append(purview_size_m)
    
    df_phi = pd.DataFrame(phis).T
    df_purview = pd.DataFrame(purview_sizes).T
    # TODO: for large sizes maybe save after every order
    if save_flag:
        
        df_phi.to_json(save_path + subject_id + '_phi_1to' + str(max_M_size) + 'rep' + str(num_mechs) + '.json')
        df_purview.to_json(save_path + subject_id + '_purview_1to' + str(max_M_size) + 'rep' + str(num_mechs) + '.json')
    
    else: 
        return df_phi, df_purview

def small_phi_evaluation_from_path(data_path, num_mechs = 100, max_M_size = 8, save_flag = False, save_path = './' ):
    cor_mat = scipy.io.loadmat(data_path)
    S_cor = cor_mat['signal']
    P_cor, _ = get_prob_matrix(S_cor)

    B_signal = binarize_signal(S_cor)
    if save_flag:
        stl = data_path.find('SUBJECT')
        small_phi_across_orders(B_signal, P_cor, num_mechs = num_mechs, 
            max_M_size = max_M_size, save_flag = save_flag, save_path = save_path, subject_id=data_path[stl:stl+13])
    else:
        return small_phi_across_orders(B_signal, P_cor, num_mechs = num_mechs, max_M_size = max_M_size)

def multi_thread_small_phi(files, num_mechs = 100, max_M_size = 8, save_flag = True, save_path = './'):
    """Runs small_phi_evaluation and creates a separate thread for each subject
    """
    threads = len(files)
    sp_input = []
    for file in files:
        sp_input.append((file, num_mechs, max_M_size, save_flag, save_path))

    pool_obj = multiprocessing.Pool()
    answer = pool_obj.starmap(small_phi_evaluation_from_path, sp_input)
    pool_obj.close()

    return answer

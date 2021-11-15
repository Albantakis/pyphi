import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy
import multiprocessing
import os
import time
import hcp_utils as hcp
import pickle

# create annealing iteration object
class anneal_iter:

   #class default constructor
  def __init__(self,iter_num,Bphi,set_list,Pmin_max_hist,iit_complex): 
          self.iter_num = iter_num
          self.Bphi = Bphi
          self.set_list = set_list
          self.Pmin_max_hist = Pmin_max_hist
          self.iit_complex = iit_complex

def bigphi_calc(P_part, percentile_thr):
    """Approximation of 'big phi' 

     Args:
        P_part (np.ndarray): A partition of the full probability matrix (P_full) from the vertice covariance matrix
        percentile_thr (float): The percentile (0.001 corresponds to top 1/1000th) in identifying the "cut" (this controls for extreme outlier correlations)

    Returns:
        Bphi (float): big phi
        Pmin_max (float): The minimum "cut", correspoding to the minimum probability among vertices (of maximum correlations identified for each vertice)
    """

    # sort vertices (here columns) by maximum correlations (i.e., maximum correlation for each vertice)
    P_sort = np.sort(P_part,1)[:,::-1]

    # control for outliers by getting 0.001 percentile 
    # LA: effectively select 0.001*#vertices column
    # LA: This can already make a significant difference in the correlation, maybe discuss
    # Pi_max = P_sort[:,(P_sort.shape[0]-1) - np.int(np.ceil(percentile_thr*P_sort.shape[0]))]
    Pi_max = P_sort[:,np.int(np.ceil(percentile_thr*P_sort.shape[0])) - 1]

    # of maximums correlations, find "cut" or minimum information partition (MIP)
    Pmin_max = Pi_max.min()
    minind = np.where(Pi_max == Pi_max.min())[0]

    # create Prob_i matrix without MIP node
    Pi_ex = Pi_max.copy()
    Pi_ex = np.delete(Pi_ex, minind, 0)

    # calclate big phi and append to list
    Bphi = np.sum(np.log10(Pi_max)) + np.log10(Pmin_max) + np.sum(np.log10(1 + Pi_ex))

    return Bphi, Pmin_max


def load_granularity(parcel_granularity_iter):
    """Loads a parcellation

     Args:
        parcel_granularity_iter (int): The ID (1-40) of a given parcellation granularity

    Returns:
        parcellation
    """
    
    this_dir, this_filename = os.path.split(__file__)
    parcel = scipy.io.loadmat(os.path.join(this_dir, "parcellations/files", 
                'parcellation_ID' + str(parcel_granularity_iter) + '_062519_Iter1_7T.mat'))
                        
    print(os.path.join(this_dir, "parcellations/files", 
                'parcellation_ID' + str(parcel_granularity_iter) + '_062519_Iter1_7T.mat'))
    
    return parcel


def initialize_annealing(parcel_granularity_iter, P_full):
    """Runs initial prior esitmation based on combinations of random parcellations

     Args:
        parcel_granularity_iter (int): The ID (1-40) of a given parcellation granularity
        P_full (np.ndarray): The full probability matrix, derived from the covariance matrix of vertices

    Returns:
        (list[anneal_iter]): List of anneal_iter objects, each list item corresponds to one parcellation ID/annealing run
    """
    
    this_dir, this_filename = os.path.split(__file__)
    parcel = scipy.io.loadmat(os.path.join(this_dir, "parcellations/files", 
                'parcellation_ID' + str(parcel_granularity_iter) + '_062519_Iter1_7T.mat'))
                        
    print(os.path.join(this_dir, "parcellations/files", 
                'parcellation_ID' + str(parcel_granularity_iter) + '_062519_Iter1_7T.mat'))
    
    # resampled parcel
    BPo = parcel['brain_parcel']

    # max # of parcels
    NPm = np.max(np.unique(BPo))
    
    # create symmetric parcellation across hemispheres
    brain_parcel = np.concatenate((BPo,BPo+NPm)).flatten().astype(np.int)
    #plt.plot(brain_parcel)
    plt.show()

    # check if brain_parcel includes zero, and shift if it does
    if np.size(np.nonzero(np.unique(brain_parcel) == 0)) > 0:
        brain_parcel = brain_parcel + 1
    D = len(brain_parcel)
    NP = len(np.unique(brain_parcel))

    # changing the order of ROI labels
    order_vec            = np.zeros(D)
    order_vec[np.arange(0,int(D)-1,2)] = np.arange(0,int(D/2))
    order_vec[np.arange(1,D,2)]   = np.arange(D/2,D)

    # reorder indicies of brain_parcel so that l/h are adjacent
    brain_parcel = brain_parcel[order_vec.astype(int)]
    # plt.plot(brain_parcel)
    # plt.show()

    # reorder indicies of probability matrix so that l/h are adjacent
    P_full_ordered  = P_full[order_vec.astype(int),:].copy()
    P_full_ordered  = P_full_ordered[:,order_vec.astype(int)]

    # create parcel combination set 2^# parcels
    lst = list(itertools.product('10', repeat=10))

    Bphi = []
    Pmin_max_hist = []

    # skip last iteration [all zeros]
    for t in range(0,2**NP-1):

        print("parcellation iteration ID: " + str(parcel_granularity_iter) + ", iter: " + str(t) + "/" + str(2**NP))
        input_set = np.zeros(D)

        # parcel combination at iteration t, account for 0-based index
        on_list = np.array([i for i,x in enumerate(lst[t]) if x=='1']) + 1
        print("parcel combination: " + str(on_list))

        # 2^NP combinations of parcels
        for ii in range(0,len(on_list)):
            input_set[np.squeeze(brain_parcel == on_list[ii])] = 1

        # indices of specific parcel combination
        ROI_ID = np.nonzero(input_set)[0]

        P_part  = P_full_ordered[ROI_ID,:]
        P_part = P_part[:,ROI_ID]

        Bphi_calc, Pmin_max_calc = bigphi_calc(P_part,0.001)

        Bphi.append(Bphi_calc)
        Pmin_max_hist.append(Pmin_max_calc)
        
    Bphi_max = np.array(Bphi).max()
    iit_complex = lst[np.where(np.array(Bphi) == Bphi_max)[0][0]]
        
    ai = anneal_iter(parcel_granularity_iter,Bphi,lst,Pmin_max_hist,iit_complex)
    
    return ai

def multi_thread_initialize_annealing(threads, P_full):
    """Initializes annealing with multipel threads 

     Args:
        P_pull (np.ndarray): Thefull probability matrix (P_full) from the vertice covariance matrix
  
    Returns:
        answer (list[anneal_iter]): list of annealing iterations
    """

    mp_input = []
    for i in range(1,threads+1):
        mp_input.append((i,P_full))

    pool_obj = multiprocessing.Pool()
    answer = pool_obj.starmap(initialize_annealing, mp_input)
    pool_obj.close()

    return answer


def run_annealing_wprior(parcel_granularity_iter, prior_dir, subid, parcel_granularity, P_full, annealing_iterations, min_prob, last_run=False):
    """Runs simulated annealing according to specific granularity iteration (2-8). It utilized the output from the prior step, 
    as a basis for monte carlo simulations to identify the optimal combinatorial solution for the IIT complex

    Keyword arguments:
        parcel_granularity_iter (int): the specific iteration ID (out of 40) that corresponds to a specific parcellation file of a given granularity
        prior_dir (str): string with directory name of prior
        subid (str): name of subject id
        min_prob (float): minimum probability for the probability of parcels based on v_prior
        P_full (np.ndarray): string with directory name of prior
    
    parcel_granularity -- the number of ROIS in parcellation file [1:8] corresponds to 10, 20, 40, 80, 160, 320, 640, 1280
    P_full -- probability matrix from neuroimaging data
    annealing_iterations -- iterations of simulated annealing
    
    """

    this_dir, this_filename = os.path.split(__file__)
    parcel = scipy.io.loadmat(os.path.join(this_dir, "parcellations/files", 
                'parcellation_ID' + str(parcel_granularity_iter) + '_062519_Iter' + str(parcel_granularity) + '_7T.mat'))
                        
    print(os.path.join(this_dir, "parcellations/files", 
                'parcellation_ID' + str(parcel_granularity_iter) + '_062519_Iter' + str(parcel_granularity) + '_7T.mat'))

    if last_run:
        v_prior = np.load(prior_dir + '/SA_random_prior_Iter' + str(parcel_granularity) +\
            '_' + subid + '.npy')
        init_size = np.load(prior_dir + '/SA_random_prior_Iter' + str(parcel_granularity) +\
            '_medianvertices_' + subid + '.npy')
    else:
        v_prior = np.load(prior_dir + '/SA_random_prior_Iter' + str(parcel_granularity - 1) +\
            '_' + subid + '.npy')

    # resampled parcel
    BPo = parcel['brain_parcel']

    # max # of parcels
    NPm = np.max(np.unique(BPo))

    brain_parcel = np.concatenate((BPo,BPo+NPm)).flatten().astype(np.int)

    if last_run:
        # v_prior (parcel is each vertice)
        brain_parcel      = np.arange(1,len(v_prior)+1)

    # check if brain_parcel includes zero, and shift if it does
    if np.size(np.nonzero(np.unique(brain_parcel) == 0)) > 0:
        brain_parcel = brain_parcel + 1

    # D is number of vertices
    D = len(brain_parcel)
    NP = len(np.unique(brain_parcel))

    # changing the order of ROI labels
    order_vec            = np.zeros(D)
    order_vec[np.arange(0,int(D)-1,2)] = np.arange(0,int(D/2))
    order_vec[np.arange(1,D,2)]   = np.arange(D/2,D)

    # order indicies of brain_parcel so that l/h are adjacent
    brain_parcel = brain_parcel[order_vec.astype(int)]
    v_prior = v_prior[order_vec.astype(int)]

    print("specific granularity: " + str(NP))

    if not last_run:
        # set P_prior to mean of ROIs in new parcel
        P_prior = np.zeros(NP)
        for i in range(1,NP+1):
            P_prior[i-1] = np.mean(v_prior[brain_parcel == i])
    else:
        P_prior = v_prior.copy()


    # edge cases (P_prior == 1) everywhere
    if len(P_prior[P_prior == 1]) == len(P_prior):
        print("where > 0.95" + str(np.where(P_prior == 1)[0]))
        P_prior[:] = 0.95

    if (min_prob > 0) and (len(P_prior[P_prior == 0]) > 0):
        # edge cases (P_prior == 0)
        print("EDGE CASE, MIN_PROB")
        P_prior[P_prior == 0] = min_prob

    # order indicies of probability matrix so that l/h are adjacent
    P_full_ordered  = P_full[order_vec.astype(int),:]
    P_full_ordered  = P_full_ordered[:,order_vec.astype(int)]

    # vertice threshold, find where v_prior vertices are greater than median.
    p_threshold = 0.5;  
    sort_index = np.argsort(v_prior)[::-1] #descending order
    V_threshold = np.zeros(D)
    V_threshold[sort_index[0:int(np.round(p_threshold*D))]] = 1

    # parcel threshold, find the probability mean of the parcels on V_threshold
    P_threshold = np.zeros(NP)
    for npi in range(1,NP+1):
        P_threshold[npi-1] = np.mean(V_threshold[brain_parcel == npi])

    init_set = np.zeros(NP)
    if not last_run:
        # initial set based on prior
        init_set[P_threshold>0.5] = 1
        input_set = np.zeros(D)
        on_list = np.nonzero(init_set)[0] + 1
        for ii in range(0,len(on_list)):
            input_set[brain_parcel==on_list[ii]] = 1
    else:
        # find the top N=init_size vertices where v_prior is highest, 
        # init_size is median size of v_prior complexes
        init_set[P_prior.argsort()[-int(init_size):][::-1]] = 1
        input_set = init_set.copy();   

    print("initialized complex size (ROIS) based on prior: " + str(len(np.nonzero(init_set)[0])))
    ROI_ID = np.nonzero(input_set)[0]

    P_part  = P_full_ordered[ROI_ID,:].copy()
    P_part = P_part[:,ROI_ID]

    # calculate big phi
    Bphi_calc, Pmin_maxcalc = bigphi_calc(P_part,0.001)

    Bphi_hist = []
    Pmin_max_hist = []
    set_hist = []

    Bphi_hist.append(Bphi_calc)
    Pmin_max_hist.append(Pmin_maxcalc)
    set_hist.append(init_set)

    # initial temperature
    T = 2
    TFACTOR = 0.9975

    for t in range(1,annealing_iterations):

        startTime = time.time()

        if t % 100 == 0:
            print("granularity: " + str(parcel_granularity) + ", parcellation iter ID: " + str(parcel_granularity_iter) + ", annealing iter: " + str(t) + "/" + str(annealing_iterations) + ", temperature: " + str(round(T, 2)))

        # run Monte Carlo simulation: each iteration identifies parcels that are more likely (based on prior)
        # to be in complex are flipped 
        A_prior = np.round(100*np.abs(init_set-P_prior))
        # plt.plot(A_prior)
        # plt.show()

        rand_ind = np.random.choice(np.arange(0,NP), 1, p = A_prior/np.sum(A_prior))

        new_set = init_set.copy() ## IMPORTANT to COPY!
        # print(rand_ind)
        if new_set[rand_ind]:
            new_set[rand_ind] = 0
        else:
            new_set[rand_ind] = 1
            
        # check if new_set is entirely zeros, and correct if so
        if np.nonzero(new_set)[0].size == 0:
            rand_ind = np.random.choice(np.arange(0,NP), 1, p = A_prior/np.sum(A_prior))
            new_set[rand_ind] = 1
            
        # calculate big_phi
        if not last_run:
            input_set = np.zeros(D)
            on_list = np.nonzero(new_set)[0] + 1
            for ii in range(0,len(on_list)):
                input_set[brain_parcel==on_list[ii]] = 1
        else:
            input_set = np.zeros(D)
            input_set[np.nonzero(new_set)[0]] = 1

        ROI_ID = np.nonzero(input_set)[0]

        P_part  = P_full_ordered[ROI_ID,:].copy()
        P_part = P_part[:,ROI_ID]

        Bphi, Pmin_maxcalc = bigphi_calc(P_part, 0.001)

        Pmin_max_hist.append(Pmin_maxcalc)

        # annealing process
        if Bphi > Bphi_hist[t-1]:
            Bphi_hist.append(Bphi)
            init_set = new_set.copy()
        else:
            delta = Bphi_hist[t-1] - Bphi
            # transition probability
            trans_p = 1/(1 + np.exp(delta/T))
            if np.random.uniform(0, 1) < trans_p:
                init_set = new_set.copy()
                Bphi_hist.append(Bphi)
            else:
                Bphi_hist.append(Bphi_hist[t-1])
                
        set_hist.append(init_set)
        # print("Bphi_hist: " + str(Bphi_hist))

        T = T*TFACTOR

        executionTime = (time.time() - startTime)
        # print('Execution time in seconds: ' + str(executionTime))

    Bphi_max = np.array(Bphi_hist).max()
    iit_complex = set_hist[np.where(np.array(Bphi_hist) == Bphi_max)[0][0]]

    if last_run:
        # reorder indices, so lh/rh aren't neighboring
        order_inv = np.zeros(D)
        order_inv[np.arange(0, int(D / 2))] = np.arange(0, int(D) - 1, 2)
        order_inv[np.arange(int(D / 2), D)] = np.arange(1, int(D), 2)

        iit_complex = set_hist[np.where(np.array(Bphi_hist) == Bphi_max)[0][0]][order_inv.astype(int)]

    ai = anneal_iter(parcel_granularity_iter,Bphi_hist,set_hist,Pmin_max_hist,iit_complex)

    if last_run:
        with open(prior_dir + subid + "_voxelwise_annealing_output.pkl","wb") as f:
            pickle.dump(ai, f)

    return ai

def create_prior(iter_list,parcel_granularity,SUBID,output_folder):
    """Creates prior from iterations of granularities (1-40). 

    Keyword arguments:
        iter_list (list[anneal_iter]): list of annealing iterations
        parcel_granularity_iter (int): the specific iteration ID (out of 40) that corresponds to a specific parcellation file of a given granularity
        subid (str): name of subject id
        output_folder (str): str with directory where to save prior
    
    """

    big_phi_combinations = []
    vscore_all = []

    # go through all iterations
    for i in range(0,len(iter_list)):

        this_dir, this_filename = os.path.split(__file__)
        parcel = scipy.io.loadmat(os.path.join(this_dir, "parcellations/files/parcellation_ID" +\
                            str(i+1) + '_062519_Iter' + str(parcel_granularity) + '_7T.mat'))
              
        # resampled parcel
        BPo = parcel['brain_parcel']
        
        # max # of parcels
        NPm = np.max(np.unique(BPo))
        
        brain_parcel = np.concatenate((BPo,BPo+NPm))

        # check if brain_parcel includes zero, and shift if it does
        if np.size(np.nonzero(np.unique(brain_parcel) == 0)) > 0:
            brain_parcel = brain_parcel + 1
            print("SHIFTING BRAIN PARCEL")
        # D is number of vertices
        
        maxed = np.array(iter_list[i].Bphi).max()
        big_phi_combinations.append(maxed)
        
        # find all candidate complexes within 5 of big phi maximum
        all_candidates = np.array(iter_list[i].Bphi)[np.array(iter_list[i].Bphi) > (maxed - 5)]
        
        overlap_all = []
        
        # for first prior, find all bigphi complexes within 5
        if parcel_granularity == 1:

            for j in range(0,len(all_candidates)):
                ind = np.where(np.array(iter_list[i].Bphi) == all_candidates[j])[0][0]
                # convert parcel numbers in complex to array, and correct for 0-based indexing
                in_complex = iter_list[i].set_list[ind]
                in_complex = np.array(list(map(int, in_complex)))
                in_complex = np.where(in_complex == 1)[0]+1
            
                # find where the parcels match the complex
                overlap = brain_parcel == in_complex 
                overlap = np.sum(overlap,axis=1)
                
                overlap_all.append(overlap)
            
            # # maximize out of all candidates (includes any parcel in at least one of multiple max complexes)
            # overlap_all = np.max(np.array(overlap_all),axis=0)
            # vscore_all.append(overlap_all)

            # add each complex within top 5
            for k in range(0,len(overlap_all)): 
                vscore_all.append(overlap_all[k])

        # parcel granularities 2-8, only add top complex
        else:
            in_complex = iter_list[i].iit_complex
            in_complex = np.array(list(map(int, in_complex)))
            in_complex = np.where(in_complex == 1)[0]+1
        
            # find where the parcels match the complex (returns multi-dimensional array)
            overlap = brain_parcel == in_complex 
            overlap = np.sum(overlap,axis=1)
            
            vscore_all.append(overlap)

    # median vertices across complexes
    median_all = np.round(np.median(np.sum(np.array(vscore_all),axis=1)))
    
    # save plot of bphi
    for ii in range(0,len(iter_list)):
        plt.ylabel("Bphi")
        if str(parcel_granularity) == 1:
            plt.xlabel("Parcel Combination Iteration (2^10)")
        else: 
            plt.xlabel("Annealing Iteration")
        plt.title("Simulated Annealing Search: Granularity Level " + str(str(parcel_granularity)))
        
        plt.plot(iter_list[ii].Bphi,linewidth=0.2)

    plt.savefig(output_folder + '/SA_random_prior_Iter' + str(parcel_granularity) + \
            '_' + SUBID + '_bphidist.png')
    # plt.show()
    plt.close() # close so plots don't overlay

    np.save(output_folder + '/SA_random_prior_Iter' + str(parcel_granularity) + \
            '_' + SUBID, np.mean(np.array(vscore_all),axis=0))
    np.save(output_folder + '/SA_random_prior_Iter' + str(parcel_granularity) + \
            '_medianvertices_' + SUBID, median_all)


def multi_thread_run_annealing_wprior(threads, prior_dir, subid, parcel_granularity, P_full, annealing_iterations,min_prob,last_run=False):
    """Runs annealing and creates a separate thread for each parcel_granularity_iter

    threads -- each thread corresponds to a parcel_granularity_iteration (40)
    """
    mp_input = []
    for i in range(1,threads+1):
        mp_input.append((i, prior_dir, subid, parcel_granularity, P_full,annealing_iterations,min_prob, last_run))

    pool_obj = multiprocessing.Pool()
    answer = pool_obj.starmap(run_annealing_wprior, mp_input)
    pool_obj.close()

    return answer
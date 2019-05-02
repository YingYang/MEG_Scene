import numpy as np
import sys, os
import scipy.io
import mne
import scipy.stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

path = [ "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Connectivity_Analysis/",
         "/home/ying/Dropbox/MEG_source_loc_proj/source_roi_cov/",
         "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"]

for path0 in path:      
    sys.path.insert(0, path0)
    
from conn_utility import (get_tf_PLV, get_corr_tf)
from ar_utility import (least_square_lagged_regression, get_tilde_A_from_At, get_dynamic_linear_coef)   
from ROI_Kalman_smoothing import (get_param_given_u)
from Stat_Utility import Fisher_method



def get_ROI_mean(subj, ROI_names, isMEG, flag_sign_flip = True):
    """
    output: data [n_trial, n_ROI, n_times]
    """
    MEGorEEG = ['EEG','MEG']
    if isMEG:
        stcdir ="/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_solution/dSPM_MEG_ave_per_im_ROI/"
    else:
        stcdir ="/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_solution/dSPM_EEG_ave_per_im_ROI/"
 
    n_ROI = len(ROI_names)
    data_list = list()
    for i in range(n_ROI):
        tmp_mat_dict = scipy.io.loadmat(stcdir  + "%s_dSPM_%s_%s.mat" %(subj,ROI_names[i], MEGorEEG[isMEG]))
        if flag_sign_flip:
            data_list.append(tmp_mat_dict['mean_sign_flip'])
        else:
            data_list.append(tmp_mat_dict['mean'])
            
        times = tmp_mat_dict['times'][0]
        
    #print times
    n_times = len(times)
    n_trials = data_list[0].shape[0]
    data = np.zeros([n_trials, n_ROI, n_times])
    for i in range(n_ROI):
        data[:,i,:] = data_list[i]
    
    # demean
    data -= np.mean(data, axis = 0)
    return data, times


#================================================================================
def get_pairwise_conn_ROI_mean(data, times, ind_array_tuple, method = "plv", 
                               bin_size = None):
    """
    Input:
        subj = Subj1
        e.g. ind_array_tuple = [[0,0,1],[1,2,2]]
        bin_size, if an integer, take the mean within a bin size of the integer
                    if None, do nothing
    Output:
        conn_list, 361 connectivity matrices for each image, each matrix is [n_pairs, n_freqs, n_times]
        freqs, array of frequence 
        times_conn, times of the conn matrix
    """
    
    if method == "plv":
        tmp, freqs, times_conn = get_tf_PLV(data, ind_array_tuple, demean = False,
                                     time_start = times.min(), sfreq  = 100.0, fmin = 5.0, fmax = 50.0)
    elif method == "corr":
        tmp_cov, tmp_corr, freqs, tstep, wsize = get_corr_tf(data, 
                                sfreq = 100.0, wsize = 160, tstep = 40)
        tmp_corr1 = np.mean(np.abs(tmp_corr),axis = 0) # note frequency 0 is half   
        tmp_corr1[0] = np.abs(tmp_corr[0,0]) 
        tmp = (tmp_corr1[:,:, ind_array_tuple[0], ind_array_tuple[1]]).transpose([2,0,1])   
        times_conn = times[0:-1:tstep]+times[tstep//2]
    if bin_size is not None:
        n_times = len(times_conn)
        tmp_new = np.reshape(tmp[:,:,0:n_times//bin_size*bin_size], 
                             [-1, len(freqs), n_times//bin_size, bin_size])
        tmp = np.mean(tmp_new, axis = -1)
        times_conn = (np.reshape(times_conn[0:n_times//bin_size*bin_size],
                                [n_times//bin_size, bin_size])).mean(axis = -1)    
    conn = tmp 
    return conn, freqs, times_conn                                
   
#============= dynamic connectivity,  AR models
def get_ar_delta_matrix(data, times, ind_array_tuple, method = "LS"):
    
    """
    data
    method ['LS','ana'] Least square regression, or fit AR and get the analitical solution
    
    """
    T = len(times)
    p = data.shape[1]
    
    Gamma0_0 = np.eye(p)
    A_0 = np.zeros([T,p,p])
    Gamma_0 = np.eye(p)
    if method == "AR1":
        # first run the non_prior version to get a global solution
        Gamma0, A, Gamma1 = get_param_given_u(data.transpose([0,2,1]), Gamma0_0, A_0, Gamma_0, 
           flag_A_time_vary = True, prior_Q0 = None, prior_A = None, prior_Q = None)
        tildeA = get_tilde_A_from_At(A)
    
    else:
        tildeA = least_square_lagged_regression(data.transpose([0,2,1]))
    
    result = np.zeros( [len(ind_array_tuple[0]), T-1,T-1])
    for i in range(len(ind_array_tuple[0])):
        pair = [ind_array_tuple[0][i], ind_array_tuple[1][i]]
        result[i] = get_dynamic_linear_coef(tildeA, pair)   
        
    return result
   
   
#==============================================================================
   
#if __name__ == "__main__":
if True:
   
    flag_sign_flip = False
    flag_plv = False
    flag_ar = True
    isMEG = False
    
    ROI_names = ['EVC','PPA','TOS','RSC','LOC','mOFC']     
    
    MEGorEEG = ['EEG','MEG']
    if isMEG:
        subj_list = range(1, 19)
    else:
        subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
        
    n_subj = len(subj_list)
    
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/"
    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/"

    nROI = len(ROI_names)
    ROI_ind = range(nROI)
    ind_array_tuple = ([0,0,0,0,0,1,1,1,],
                       [1,2,3,4,5,2,3,5,])
    n_pairs = len(ind_array_tuple[0])
    offset = 0.04 if isMEG else 0.02
    method1 = "plv"
    method2 = "AR1"
    #method2 = "LS"
    bin_size = None
    n1,n2 = 2,4
    
    #======================= PLV ================================================
    if flag_plv:
        n_freq, n_times = 45, 109
        conn_all = np.zeros([n_subj, n_pairs, n_freq, n_times])
        # === test
        for i in range(n_subj):
            subj = "Subj%d" %(subj_list[i])
            data, times = get_ROI_mean(subj, ROI_names, isMEG, flag_sign_flip = flag_sign_flip)
            times -= offset
            data = data[:,:,0:n_times]
            times = times[0:n_times]
            conn_all[i], freqs, times_conn = get_pairwise_conn_ROI_mean(data, times, 
                                   ind_array_tuple, method = method1, 
                                   bin_size = None)
    
        mat_name =  outdir + "conn_%s_%s_signflip%d.mat" %(MEGorEEG[isMEG], method1, flag_sign_flip)
        scipy.io.savemat(mat_name, dict(conn_all = conn_all, freqs = freqs, times_conn = times_conn))
        
        #======== visulializing the results
        conn_mat_all_subj = conn_all
        times_ms = times_conn*1000.0
        # general sum across all subjects, with no tests at all
        mean_conn_mat = conn_mat_all_subj.mean(axis = 0)
        for i in range(n_subj):
            tmp = conn_mat_all_subj[i]
            tmp = np.transpose(tmp,[2,0,1]) - np.mean(tmp[:,:,times_conn<= 0.0], axis = -1)
            conn_mat_all_subj[i] = tmp.transpose([1,2,0])
        conn_mat_no_baseline = conn_mat_all_subj 
        T_increase = np.mean(conn_mat_no_baseline, axis = 0)/np.std(conn_mat_no_baseline, axis = 0)*np.sqrt(n_subj)
        # do Fisher's method here?
        # [right sided, left sided]
        p_T = 2*(1-scipy.stats.t.cdf(np.abs(T_increase),df = n_subj -1))
         
        # try BH-FDR
        reject, _ = mne.stats.fdr_correction(p_T.ravel(),alpha = 0.05)
        if reject.sum() > 0:
            T_thresh =  (np.abs(T_increase).ravel()[reject]).max()
     
        data_list = [mean_conn_mat, conn_mat_no_baseline.mean(axis = 0), T_increase]
        data_names = ['mean_conn', 'mean_conn_no_baseline', 'mean_T']
        vmin_seq = [0,-0.1,-6]
        vmax_seq = [0.5,0.1, 6]
        figsize = (19,8)
        for l in range(len(data_list)):
            fig = plt.figure(figsize = figsize)
            for k in range(len(ind_array_tuple[0])):
                _= plt.subplot(n1,n2,k+1)
                im=plt.imshow(data_list[l][k], aspect = "auto", interpolation = "none",
                           extent = [times_ms[0],times_ms[-1], freqs[0], freqs[-1]],
                           origin = "lower",
                           vmin = vmin_seq[l], vmax = vmax_seq[l]);
                #_=plt.colorbar(); 
                _=plt.xlabel(ROI_names[ind_array_tuple[1][k]] +"-"+ROI_names[ind_array_tuple[0][k]])
                _=plt.ylabel("frequence")
                #_=plt.axis('equal')
            cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.4])
            fig.colorbar(im, cax=cbar_ax)
            #plt.tight_layout()
            plt.savefig(fig_outdir + "conn_%s_%s_signflip%d_%s.pdf"\
               %(MEGorEEG[isMEG], method1, flag_sign_flip, data_names[l]))
        
        plt.close('all')
            
        # permutation tests
        p_thresh = 0.1 #/n_pairs
        fig = plt.figure(figsize = figsize)
        for k in range(n_pairs):
            _= plt.subplot(n1,n2,k+1)
            tmp_data = conn_mat_no_baseline[:,k,:,:]
            Tobs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_1samp_test(tmp_data, threshold = None)
            mask = np.zeros(Tobs.shape, dtype = np.bool)
            for l in range(len(clusters)):
                if cluster_pv[l] <= p_thresh:
                    mask[clusters[l]] = True
                
                
            Tobs[mask == False] = 0.0       
            im=plt.imshow(Tobs, aspect = "auto", interpolation = "none",
                       extent = [times_ms[0],times_ms[-1],freqs[0], freqs[-1]],
                       origin = "lower", vmin = -6 , vmax = 6 ); 
            _=plt.title(ROI_names[ind_array_tuple[0][k]]+" "
             +ROI_names[ind_array_tuple[1][k]] )
            _=plt.xlabel('time (ms)')
            _=plt.ylabel('frequency (Hz)')
            
        cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.4])
        fig.colorbar(im, cax=cbar_ax)
        plt.savefig(fig_outdir + "conn_%s_%s_signflip%d_%s.pdf"\
               %(MEGorEEG[isMEG], method1, flag_sign_flip, "permtest"))
        plt.close()
    #======================================================================================
    
    if flag_ar:
        outdir0 = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/roi_ks/"
        btstrap_path = outdir0 + "btstrp_seq.mat"
        if os.path.isfile(btstrap_path):
            print "bootstrap seq exits!"
            pass
        else:
            n_im = 362
            if not os.path.isfile(btstrap_path):
                B = 50
                original_order = (np.arange(0,n_im)).astype(np.int)
                bootstrap_seq = np.zeros([B+1, n_im], dtype = np.int)
                bootstrap_seq[0] = original_order
                for l in range(B):
                    bootstrap_seq[l+1] = (np.random.choice(original_order, n_im)).astype(np.int)
                scipy.io.savemat(btstrap_path, dict(bootstrap_seq = bootstrap_seq))
        #=================== load the btstrap sequence ================================
        # all subjects shared the same one?
        B = 30
        bootstrap_dict = scipy.io.loadmat(btstrap_path)   
        bootstrap_seq = bootstrap_dict['bootstrap_seq'].astype(np.int) 
        bootstrap_seq = bootstrap_seq[0:B+1,:]
        
        n_times = 109 
        Delta_all = np.zeros([n_subj, B+1, n_pairs,n_times-1 , n_times-1])
        for i in range(n_subj):
            subj = "Subj%d" %(subj_list[i])
            data, times = get_ROI_mean(subj, ROI_names, isMEG, flag_sign_flip = True)
            times = times -offset
            times = times[0:n_times]
            t0 = time.time()
            for l in range(B+1):
                tmp_data = data[bootstrap_seq[l],:,0:n_times]
                Delta_all[i,l] = get_ar_delta_matrix(tmp_data, times, ind_array_tuple, method = method2)
        
            print time.time()-t0
        mat_name =  outdir + "conn_%s_%s_signflip%d.mat" %(MEGorEEG[isMEG], method2, flag_sign_flip)
        scipy.io.savemat(mat_name, dict(times = times, Delta_all = Delta_all)) 


        #============= visulizing delta all =======================
        mat_name =  outdir + "conn_%s_%s_signflip%d.mat" %(MEGorEEG[isMEG], method2, flag_sign_flip)
        mat_dict = scipy.io.loadmat(mat_name) 

        Delta_all = mat_dict['Delta_all']
        times = mat_dict['times'][0]


        Delta_all0 = Delta_all[:,0,:,:]
        Delta_all_Z = Delta_all[:,0,:,:]/np.std(Delta_all[:,1::,:,:], axis = 1)
        Delta_all_Z[np.isnan(Delta_all_Z)]= 0
        # Z-score tests
        Delta_all_logp = -np.log10(2*(1.0-scipy.stats.norm.cdf(np.abs(Delta_all_Z))))
        Delta_all_logp[np.isinf(Delta_all_logp)] = 3.0
        Fisher_logp = np.zeros([n_pairs, n_times-1, n_times-1])
        for k in range(n_pairs):
            for i in range(n_times-1):
                for j in range(n_times-1):
                    Fisher_logp[k,i,j] = -np.log10(Fisher_method(10.0**(-Delta_all_logp[:,k,i,j])))
                
        mean_Delta_all = (Delta_all_Z).mean(axis = 0)
        T_Delta_all =(Delta_all_Z).mean(axis = 0)/np.std((Delta_all_Z), axis = 0)*np.sqrt(n_subj)
        
        times_ms = times*1000.0
        data_list = [Fisher_logp, Delta_all_logp.mean(axis =0),]  #, mean_Delta_all T_Delta_all]
        data_names = ['Fisherlog10p','meanLog10p',] #, 'mean_delta' 'TDelta']
        vmin_seq = [-np.log10(0.05/n_pairs/((n_times-1)**2)),0, 0, 0]
        vmax_seq = [10,5, 5, 10]
        figsize = (19,8)
        for l in range(len(data_list)):
            fig = plt.figure(figsize = figsize)
            for k in range(len(ind_array_tuple[0])):
                _= plt.subplot(n1,n2,k+1)
                im=plt.imshow(data_list[l][k], aspect = "auto", interpolation = "none",
                           extent = [times_ms[0],times_ms[-1], times_ms[0], times_ms[-1]],
                           origin = "lower",
                           vmin = vmin_seq[l], vmax = vmax_seq[l]);
                #_=plt.colorbar(); 
                _=plt.xlabel(ROI_names[ind_array_tuple[1][k]] +" time (ms)")
                _=plt.ylabel(ROI_names[ind_array_tuple[0][k]] +" time (ms)")
                #_=plt.axis('equal')
            cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.4])
            fig.colorbar(im, cax=cbar_ax)
            #plt.tight_layout()
            plt.savefig(fig_outdir + "conn_%s_%s_signflip%d_%s.pdf"\
               %(MEGorEEG[isMEG], method2, flag_sign_flip, data_names[l]))
       
        
    
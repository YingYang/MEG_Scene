# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import time

import matplotlib.pyplot as plt
import scipy.io
import scipy.stats

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)


meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/MEEG_comp/"


flag_run = False
flag_plot = True


subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
n_subj = len(subj_list)

#offset_seq = np.arange(0.01, 0.05, 0.01)
offset_seq = [0.02]
n_offset = len(offset_seq)

n_im = 362


if flag_run:

    # with permutation tests:
    n_perm = 40
    perm_seq = np.zeros([n_perm, n_im], dtype = np.int)
    orig_seq = range(0, n_im)
    for i in range(n_perm):
        perm_seq[i] = np.random.permutation(orig_seq)
     
    perm_seq_aug = np.zeros([n_perm+1, n_im], dtype =np.int)
    perm_seq_aug[0] = orig_seq
    perm_seq_aug[1::] = perm_seq   
    
    
    corr_ts = np.zeros([n_subj, n_offset], dtype = np.object)
    corr_ts_perm = np.zeros([n_subj, n_offset], dtype = np.object)
    corr_ts_logpval = np.zeros([n_subj, n_offset], dtype = np.object)
    common_times_all = np.zeros(n_offset, dtype = np.object)
    
    
    for l in range(n_offset):
        offset = offset_seq[l]
        for i in range(n_subj):   
            subj = "Subj%d" %subj_list[i]
            print subj, l
            meg_fname_suffix = "1_110Hz_notch_ica_MEEG_match_ave_alpha15.0_no_aspect"
            MEG_ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,meg_fname_suffix)
            data = scipy.io.loadmat(MEG_ave_mat_path)
            MEG_times = np.round(data['times'][0]-offset, decimals = 2)
            MEG_data = data['ave_data']
            MEG_picks_all = data['picks_all'][0]
            MEG_data -= np.mean(MEG_data,axis = 0)
            del(data)
    
            eeg_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_MEEG_match_ave_alpha15.0_no_aspect"
            EEG_ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,eeg_fname_suffix)
            data = scipy.io.loadmat(EEG_ave_mat_path)
            EEG_times = np.round(data['times'][0], decimals = 2)
            EEG_data = data['ave_data']
            EEG_picks_all = data['picks_all'][0]
            EEG_data -= np.mean(EEG_data,axis = 0)
            del(data)
            
            #==================================================================================
            # take the intersection of time points
            tmin = np.round(max(MEG_times[0], EEG_times[0]),decimals = 2)
            tmax = np.round(min(MEG_times[-1], EEG_times[-1]),decimals = 2)
            common_times  = np.arange(tmin, tmax, 0.01)
            MEG_in_common_time_id = [l0 for l0 in range(len(MEG_times)) if MEG_times[l0] >= tmin and MEG_times[l0]<= tmax]
            EEG_in_common_time_id = [l0 for l0 in range(len(EEG_times)) if EEG_times[l0] >= tmin and  EEG_times[l0] <=tmax]
            MEG_data = MEG_data[:,:,MEG_in_common_time_id]
            EEG_data = EEG_data[:,:,EEG_in_common_time_id]
            
    
            
            # compute the RSM correlation
            n_times = len(common_times)
            tmp_corr_ts = np.zeros([n_perm+1, n_times])
            n_im = 362
            mask = np.ones([n_im, n_im], dtype = np.bool)
            mask = np.triu(mask, 1)
            #ktau = np.zeros([n_perm+1, n_times])
            #ktau_p = np.zeros([n_perm+1, n_times])
            
            for j in range(n_perm+1):
                for t in range(n_times):
                    tmp_MEG_data = MEG_data[:,:,t].copy()
                    tmp_EEG_data = EEG_data[:,:,t].copy()
                    tmp_MEG_data = tmp_MEG_data[perm_seq_aug[0]]
                    tmp_EEG_data = tmp_EEG_data[perm_seq_aug[j]]
                    corr1 = np.corrcoef(tmp_MEG_data)
                    corr2 = np.corrcoef(tmp_EEG_data)
                    tmp1 = corr1[mask>0]
                    tmp2 = corr2[mask>0]
                    tmp_corr_ts[j,t] = np.corrcoef(tmp1, tmp2)[0,1]
                    tmp_rank1 = np.argsort(tmp1)
                    tmp_rank2 = np.argsort(tmp2)
                    # I should not use rank, funny
                    #ktau[j,t],ktau_p[j,t] = scipy.stats.kendalltau(tmp_rank1, tmp_rank2)
                    #ktau[j,t],ktau_p[j,t] = scipy.stats.kendalltau(tmp1, tmp2)
                    
            common_times_all[l] = common_times
            corr_ts[i,l] = tmp_corr_ts[0]
            corr_ts_perm[i,l] = tmp_corr_ts[1:]
            tmp_p_val = np.mean(tmp_corr_ts[1:]-tmp_corr_ts[0,:]>0, axis = 0)
            tmp_p_val[tmp_p_val ==0] = 1.0/corr_ts.shape[0]
            corr_ts_logpval[i,l] = -np.log10(tmp_p_val)
         
            
     
    corr_ts_mat = np.zeros(n_offset, dtype = np.object)
    corr_ts_logpval_mat =  np.zeros(n_offset, dtype = np.object)
    
    for l in range(n_offset):
        tmp_corr_ts_mat = np.zeros([n_subj, len(common_times_all[l])])
        tmp_corr_ts_logpval_mat = np.zeros([n_subj, len(common_times_all[l])])
        for i in range(n_subj):
            tmp_corr_ts_mat[i] = corr_ts[i,l]
            tmp_corr_ts_logpval_mat[i] = corr_ts_logpval[i,l]
        corr_ts_mat[l] = tmp_corr_ts_mat   
        corr_ts_logpval_mat[l] = tmp_corr_ts_logpval_mat
        
    
    mat_dict = dict(corr_ts_mat = corr_ts_mat, corr_ts_logpval_mat = corr_ts_logpval_mat, 
                    common_times_all =common_times_all,
                    corr_ts = corr_ts, corr_ts_perm = corr_ts_perm,
                    subj_list = subj_list)
                    
    
    scipy.io.savemat(mat_name + "RSM.mat", mat_dict)  

#================================================================================

if flag_plot:
    
    
    
    
    
    n_offset = 1
    common_offset = 0.02
    #============ single offset, hard coded do the permutation tests ===================
    mat_dict = scipy.io.loadmat(mat_name + "RSM.mat")
    corr_ts_mat = mat_dict['corr_ts_mat'][0,0]
    corr_ts_logpval_mat = mat_dict['corr_ts_logpval_mat'][0,0]
    common_times_all =  mat_dict['common_times_all'][0,0] [0]
    corr_ts_perm = mat_dict['corr_ts_perm'][:,0]
    n_perm, n_times = corr_ts_perm[0].shape
    
    corr_ts_perm_mat = np.zeros([n_subj, n_perm, n_times])
    for i in range(n_subj):
        corr_ts_perm_mat[i,:,:] = corr_ts_perm[i]
        
    
    # compute the averaged
    ave_corr_ts= corr_ts_mat.mean(axis = 0)
    ave_corr_ts_perm = corr_ts_perm_mat.mean(axis = 0)
    
    times_in_ms = (common_times_all-common_offset)*1000.0
    
    plt.figure(figsize = (8,4))
    ax = plt.subplot(1,1,1)
    _= ax.plot(times_in_ms, ave_corr_ts)
    alpha = 0.05
    percentile_list = [alpha/2.0*100.0, (1-alpha/2.0)*100]
    lb,ub = np.percentile(ave_corr_ts_perm, percentile_list, axis = 0)
    _ = ax.fill_between(times_in_ms, ub, lb, facecolor='b', alpha=0.4) 

    # permutation test, two sided  
    sys.path.insert(0, "/home/ying/Dropbox/MEG_source_loc/Face_Learning_Data_Ana/") 
    from Stat_Utility import excursion_perm_test_1D
    thresh = np.median(ub)
    clusters, integral, p_val_clusters = excursion_perm_test_1D( ave_corr_ts, ave_corr_ts_perm, thresh, tail = 0)

    print clusters, p_val_clusters
    cluster_p_thresh = 0.05
    for i_c, c in enumerate(clusters):
            c = c[0]
            if p_val_clusters[i_c] <= cluster_p_thresh:
                _ = ax.axvspan(times_in_ms[c.start], times_in_ms[c.stop - 1],
                                    color='r', alpha=0.2)
                #_ = plt.text(times_in_ms[c.start],-0.01,('p = %1.3f' %p_val_clusters[i_c]))
    plt.xlabel('time (ms)')
    plt.ylabel('RSM correlation')  
    plt.xlim(-100.0, 900.0)  
    plt.grid('on')                                         
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/meg_eeg_sensor/"            
    plt.savefig(fig_outdir + "RSM_corr_ave.pdf")
    
    
        
    
    """
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/meg_eeg_sensor/"
    data_list = [corr_ts_mat,corr_ts_logpval_mat]
    data_name = ['corr','logpval']
    for j in range(2):
        plt.figure()
        vmin = -0.1
        vmax = 0.8
        data = data_list[j]
        for l in range(n_offset): 
            plt.subplot(n_offset,1,l+1)
            plt.imshow(data[0,l], aspect = "auto",
                           interpolation = "none", 
                           extent = [common_times_all[0,l][0,0], common_times_all[0,l][0,-1], 0,n_subj],
                           origin = "lower",
                           vmin = vmin, vmax = vmax)
            plt.colorbar() 
            plt.title("offset %1.2f" % offset_seq[l])          
        
        plt.figure()
        for l in range(n_offset):
            plt.plot(common_times_all[0,l][0], np.mean(data[0,l],axis = 0), '*-', lw = 2)
            plt.legend(offset_seq) 
            
        plt.xlabel('time (ms)')
        plt.ylabel('RSM')
        plt.grid('on')         
        plt.savefig(fig_outdir + "RSM_%s_ave.pdf" %data_name[j])   
        

    import mne
    import sys
    path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
    sys.path.insert(0, path0)
    from Stat_Utility import bootstrap_mean_array_across_subjects
    
    l = 0
    times = common_times_all[0,l][0]
    corr_ts = corr_ts_mat[0,l]
    logp_corr_ts = corr_ts_logpval_mat[0,l]
    logp_corr_ts = (logp_corr_ts.T - np.mean(logp_corr_ts[:, times<0], axis = 1)).T
    
    data_list = [corr_ts, logp_corr_ts]
    data_names = ['RSM_corr', '-log10_p_perm']
    for j in range(2):
        tmp = bootstrap_mean_array_across_subjects(data_list[j])
        tmp_mean = tmp['mean']
        tmp_se = tmp['se']
        ub = tmp['ub']
        lb = tmp['lb']              
        # only consider the time after zero
        threshold = 1
        Tobs, clusters, p_val_clusters, H0 = mne.stats.permutation_cluster_1samp_test(
            data_list[j], threshold,tail = 0)
        print clusters, p_val_clusters
        cluster_p_thresh = 0.05
        plt.figure(figsize=(6.5,5))
        ax = plt.subplot(1,1,1)
        ax.plot(times, np.mean(data_list[j], axis = 0))
        tmp_window = list()
            for i_c, c in enumerate(clusters):
                c = c[0]
                if p_val_clusters[i_c] <= cluster_p_thresh:
                    _ = ax.axvspan(times[c.start], times[c.stop - 1],
                                        color='r', alpha=0.2)
                    _ = plt.text(times[c.start],0.1, 
                                                     ('p = %1.3f' %p_val_clusters[i_c])) 
                    tmp_window.append(dict(start = c.start, stop = c.stop, p = p_val_clusters[i_c]))                               
            _ = ax.fill_between(times, ub, lb, facecolor='b', alpha=0.4)     
        # temporal bootstrap and test against zero
        plt.xlabel('time (s)')
        plt.tight_layout()
        plt.ylabel(data_names[j])
        plt.grid()
        plt.tight_layout()
        plt.savefig(fig_outdir + "RSM_lag_20ms_%s_ave.pdf" %data_name[j]) 
    """

#=========================== noise ceiling? ======================================




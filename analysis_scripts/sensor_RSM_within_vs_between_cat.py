import numpy as np
import scipy.io
import scipy.stats
import scipy.spatial
from copy import deepcopy

path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
import sys
sys.path.insert(0,path0)
from Stat_Utility import bootstrap_mean_array_across_subjects
import matplotlib.pyplot as plt 
import mne

def sensor_rdm_within_vs_between_cat(data, metric = "euclidean"):
    """
        eg: 
        subj = "Subj1"
        fname_suffix = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/" \
            + "%s/%s_1_50Hz_raw_ica_window_50ms_first_100percent_ave.mat" %(subj, subj)
       
    """
    
    n_times = data.shape[2]
    n_im = 362
    # within between mask: hard-coded!!!
    mask = np.ones([n_im,n_im], dtype = np.int)
    mask = np.triu(mask, k = 1)
    for i in range(n_im//2):
        mask[2*i,2*i+1] = 2
    
    # we want it to be positive
    within_between_ratio = np.zeros(n_times) 
    for i in range(n_times):
        
        tmp = scipy.spatial.distance.squareform( 
                  scipy.spatial.distance.pdist(data[:,:,i],metric=metric))
        within_between_ratio[i] =  np.mean(tmp[mask == 2])/np.mean(tmp[mask == 1])
        
    return within_between_ratio
    
#==========================================================================
if __name__ == '__main__':
    
    isMEG = 0;
    if isMEG:
        subj_list = range(1,19)
        n_times = 110
    else:
        subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
        n_times = 109
    n_subj = len(subj_list)
    ratio_full = np.zeros([n_subj, n_times])
    metric = "correlation"
    
    meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
    eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
    fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    
    
    MEGorEEG = ["EEG","MEG"]
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        if isMEG:
            ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
        else:
            ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
        
        mat_dict = scipy.io.loadmat(ave_mat_path)
        data = mat_dict['ave_data']
        data = data - np.mean(data, axis = 0)
        times = mat_dict['times'][0][0:n_times]            
        within_between_ratio = sensor_rdm_within_vs_between_cat(data, metric = metric)
        ratio_full[i] = within_between_ratio[0:n_times]
        del(data, mat_dict)

    
    #==================== save mat ===========================================
    mat_dict = dict(ratio_full = ratio_full, times = times)
    mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_dependence/"\
               + "Subj_pooled_rdm_within_vs_between_cat_%s_%s.mat" %(metric, MEGorEEG[isMEG])
    scipy.io.savemat(mat_name, mat_dict)
    
    offset = 0.04
    times_in_ms = (times - offset)*1000.0
    log_ratio_full = np.log10(ratio_full)

    tmp = bootstrap_mean_array_across_subjects(log_ratio_full)
    tmp_mean = tmp['mean']
    tmp_se = tmp['se']
    ub = tmp['ub']
    lb = tmp['lb']
   
    
    baseline_time_ind = times< 0
    baseline_mean = np.mean(log_ratio_full[:, baseline_time_ind], axis = 1)
    logp_val_time_selected_no_baseline = (log_ratio_full.T - baseline_mean).T
    # only consider the time after zero
    threshold = None
    Tobs, clusters, p_val_clusters, H0 = mne.stats.permutation_cluster_1samp_test(
    logp_val_time_selected_no_baseline, threshold,tail = -1)
    print clusters, p_val_clusters
    
    cluster_p_thresh = 0.05

    plt.figure(figsize=(4.5,3))
    ax = plt.subplot(1,1,1)
    ax.plot(times_in_ms, np.mean(log_ratio_full, axis = 0))
    for i_c, c in enumerate(clusters):
        c = c[0]
        if p_val_clusters[i_c] <= cluster_p_thresh:
            _ = ax.axvspan(times_in_ms[c.start], times_in_ms[c.stop - 1],
                                color='r', alpha=0.1)
            _ = plt.text(times_in_ms[c.start],0.25, 
                                             ('p = %1.3f' %p_val_clusters[i_c])) 
    _ = ax.fill_between(times_in_ms, ub, lb, facecolor='b', alpha=0.1)     
  
    # temporal bootstrap and test against zero
  
    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.title(metric + "_RDM_within_between_ave")
    plt.ylabel('log10_ratio')
    plt.grid()
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_rsm/"
    plt.savefig(fig_outdir + "%s_within_between_ratio_%s.pdf" %(metric, MEGorEEG[isMEG]))
    


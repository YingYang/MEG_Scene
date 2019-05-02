# -*- coding: utf-8 -*-
import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io

import sys
path0 = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/analyze_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/analyze_scripts/Util/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path0)


from RSM import get_rsm_correlation
from Stat_Utility import excursion_perm_test_1D

meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
isMEG = True
if isMEG:
    Subj_list = range(1,19)
else:
    Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
n_Subj = len(Subj_list)
mat_file_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_dependence/"
offset = 0.04 if isMEG else 0
    
n_im = 362
n_perm = 50
orig_seq = np.arange(0,n_im)
perm_seq = np.zeros([n_perm, n_im])
for k in range(n_perm):
    perm_seq[k] = np.random.permutation(orig_seq)
    perm_seq = perm_seq.astype(np.int)

for X_id in [ 0,1,2]:
    if X_id in [0,1,2]:
        # load the data X
        mat_data = scipy.io.loadmat('/home/ying/dropbox_unsync/MEG_scene_neil/PTB_Experiment/selected_image_second_round_data.mat');
        neil_attr_score = mat_data['attr_score']
        neil_low_level = mat_data['low_level_feat']
        is_high = mat_data['is_high'][:,0]
        neil_scene_score = mat_data['scene_score']
        
        if X_id == 0:
            X0 = neil_attr_score 
            feat_name = "neil_attr"
        elif X_id == 1:
            X0 = neil_low_level 
            feat_name = "neil_low"
        elif X_id == 2:
            X0 = neil_scene_score 
            feat_name = "neil_scene"

    #mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/AlexNet_Features/AlexNetImageFeatures_%s_%s.mat" %(layer, feature_suffix)
    #X0= mat_data['data']
    # for each feature, always demean them first
    X = X0- np.mean(X0, axis = 0)    
    Result_list = list()
    for i in range(n_Subj):
        subj = "Subj%d" %Subj_list[i]
        if isMEG:
            ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)            
        else:
            ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix) 
        ave_mat = scipy.io.loadmat(ave_mat_path)
        ave_data = ave_mat['ave_data']
        #compute correlation
        result_rdm = get_rsm_correlation(ave_data, X, n_perm = n_perm,
                         perm_seq = perm_seq, metric = "correlation", demean = True, alpha = 0.05)   
        result_rdm['times'] = ave_mat['times']
        if isMEG:
            mat_name = mat_file_out_dir + "%s_MEG_result_rdm_%s.mat" %(subj, feat_name)
        else:
            mat_name = mat_file_out_dir + "%s_EEG_result_rdm_%s.mat" %(subj, feat_name)
        scipy.io.savemat(mat_name, result_rdm)
        Result_list.append(result_rdm)
    
    corr_ts = np.zeros([n_Subj, 110])
    corr_ts_perm = np.zeros([n_Subj, n_perm, 110])
    for i in range(n_Subj):  
        corr_ts[i] = Result_list[i]['corr_ts']
        corr_ts_perm[i] = Result_list[i]['corr_ts_perm']
      
    # =========== plot individual subjects ====================================            
    plt.figure(figsize = (10,8))
    ymin, ymax  = -0.015, 0.12
    times_in_ms = (result_rdm['times']-offset)*1000.0
    m1,m2 = 4,4
    for i in range(n_Subj):
        if True:
            result_rdm = Result_list[i]
            ax = plt.subplot(m1,m2,i+1)
            plt.plot(times_in_ms, result_rdm['corr_ts'])
            plt.plot(times_in_ms, np.zeros(result_rdm['times'].size))
            plt.fill_between(times_in_ms, result_rdm['null_range'][0],result_rdm['null_range'][1],alpha = 0.2)
            plt.ylim( ymin, ymax)
            plt.title("Subj%d" % Subj_list[i])
            plt.xlabel('time (ms)')
    plt.tight_layout()
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_rdm/"
    plt.savefig(fig_outdir + "Subj_indiv_%s.pdf" %feat_name)
    
    
    #============= plot group mean ============================================
    corr_ts_mean = np.mean(corr_ts,axis = 0)
    corr_ts_per_mean = np.mean(corr_ts_perm, axis = 0)
    plt.figure(figsize=(4.5,3))
    ax = plt.subplot(1,1,1)
    plt.grid()
    plt.plot(times_in_ms, corr_ts_mean)
    alpha = 0.05
    percentile = np.percentile(corr_ts_per_mean, [alpha*100.0/2.0, (1.0-alpha/2.0)*100.0], axis = 0 )
    plt.fill_between(times_in_ms, percentile[0],percentile[1],alpha = 0.2)
    # mark the significant ones with excursion tests
    threshold = max(np.max(np.abs(percentile[0])), np.max(np.abs(percentile[1])))
    print "X_id and threshold"
    print X_id, threshold
    cluster, _, p_val =excursion_perm_test_1D( corr_ts_mean, corr_ts_per_mean, 
                            threshold = threshold , tail = 0)
    cluster_p_thresh = 0.05                        
    if len(cluster) > 0:
        l = 0
        for k in range(len(p_val)):
            c = cluster[k][0]
            if p_val[k] <= cluster_p_thresh:
                _ = ax.axvspan(times_in_ms[c.start], times_in_ms[c.stop - 1],
                                        color='r', alpha=0.1)
                if np.mod(l,2):
                    ratio = 0.75                            
                else:
                    ratio = 0.65
                
                p_val_to_show = p_val[k] if p_val[k]!=0 else 1.0/n_perm
                _ =  plt.text(times_in_ms[c.start], ymax *ratio, 
                             ('%1.3f' %p_val_to_show))
                l += 1 
    plt.ylim( ymin, ymax)
    plt.xlabel('time (s)')
    plt.tight_layout()
    plt.savefig(fig_outdir+"Subj1_9_ave_%s.pdf" %feat_name)
    
    plt.close('all')
    
    
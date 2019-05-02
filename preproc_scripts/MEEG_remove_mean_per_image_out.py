# -*- coding: utf-8 -*-
"""
For each trial, remove the mean across all repetitions of images, 
then they can be used for "noise-correlation type of analysis"
"""

import numpy as np
import scipy.io
if False:
    # MEG
    isMEG = 1
    Subj_list = range(1,10)
    data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
    #epoch_dir = "epoch_raw_data"
    #save_name_suffix = "1_50Hz_raw_ica_window_%dms" %(0.05*1000)
    
    # for tsssed data
    epoch_dir = "epoch_tsssmc_data"
    save_name_suffix = "1_50Hz_tsssmc_ica_window_%dms" %(0.05*1000)
    
if True:
    # EEG
    isMEG = 0
    Subj_list = [1,2,3,4,5,6,7,8,10]
    data_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
    epoch_dir = "epoch_raw_data"
    save_name_suffix = "1_50Hz_raw_ica_window_50ms"



#====================================================================================
n_subj = len(Subj_list)
for j in range(n_subj):
    
    if isMEG:
        subj = "Subj%d" %Subj_list[j] 
    else:
        subj = "Subj%d_EEG" %Subj_list[j] 
    
    mat_name_in = data_dir + "%s/%s/%s_%s_all_trials.mat" %(epoch_dir, subj, subj, save_name_suffix)
    mat_dict_in = scipy.io.loadmat(mat_name_in)
    epoch_mat_no_repeat = mat_dict_in['epoch_mat_no_repeat']
    im_id_no_repeat = mat_dict_in['im_id_no_repeat'][0]
    picks_all = mat_dict_in['picks_all'][0]
    bad_trials_no_repeat = mat_dict_in['bad_trials_no_repeat'][0]> 0
    times = mat_dict_in['times'][0]
    
    # remove bad channels
    
    n_times = epoch_mat_no_repeat.shape[2]
    im_id1 = im_id_no_repeat[bad_trials_no_repeat == 0]
    epoch_mat1 = epoch_mat_no_repeat[bad_trials_no_repeat == 0]
    n_im = len(np.unique(im_id_no_repeat))
    epoch_mat_residual = np.zeros(epoch_mat1.shape)
    
    im_id1 -= 1 # im_id should start from zero
    for i in range(n_im):
        tmp = im_id1 == i
        if np.sum(tmp) >0:
            tmp_data = epoch_mat1[tmp,:,:]
            tmp_mean = np.mean(tmp_data, axis = 0)
            epoch_mat_residual[tmp,:,:] = tmp_data - tmp_mean    
        else:
            print subj, j
            print "im %d not found" %i
    mat_dict_out = dict(epoch_mat_residual = epoch_mat_residual, times = times,
                        im_id = im_id1+1, picks_all = picks_all)
    mat_name_out = data_dir + "%s/%s/%s_%s_residual_all_trials.mat" \
                  %(epoch_dir, subj, subj, save_name_suffix)
    scipy.io.savemat(mat_name_out, mat_dict_out, oned_as = "row")              
    

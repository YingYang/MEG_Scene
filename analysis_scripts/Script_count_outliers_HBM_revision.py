#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 21:41:34 2019

@author: yy
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import mne
mne.set_log_level('WARNING')
import numpy as np
import scipy
import scipy.io
import scipy.stats


#===================================================================================
#Subj_list = range(1,14)
#Subj_list = range(14,16)
Subj_list = range(1,19)
#====================================================================================

# smoothed
#window_length = 0.05
#save_name_suffix = "1_50Hz_raw_ica_window_%dms" %(window_length*1000)
# unsmoothed
save_name_suffix = "1_110Hz_notch_ica"

data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
data_root_dir =data_root_dir0 +"MEG_preprocessed_data/MEG_preprocessed_data/"
epoch_dir = data_root_dir + "epoch_raw_data/"
tmin, tmax = -0.1, 1.0
n_subj = len(Subj_list)

# used to label bad trials, discard the trials where the range > 3std
alpha = 15 

n_bad_trials = np.zeros(18)
proportion_bad_trials = np.zeros(18)
#==============================================================================
for j in range(n_subj):
    subj = "Subj%d" %Subj_list[j]
    print subj
    mat_name = epoch_dir + "%s/%s_%s_all_trials.mat" %(subj, subj, save_name_suffix)
    mat_dict = scipy.io.loadmat(mat_name)
    epoch_mat_no_repeat = mat_dict['epoch_mat_no_repeat']
    im_id_no_repeat = mat_dict['im_id_no_repeat'][0]
    picks_all = mat_dict['picks_all'][0]
    times = mat_dict['times'][0]
        
    n_channels  = len(picks_all)
    n_times = epoch_mat_no_repeat.shape[2]
    
    n_trials = epoch_mat_no_repeat.shape[0]
    # each channel, each trial, the difference of two extremes in the time window
    ranges_each_trial = np.max(epoch_mat_no_repeat, axis = 2) - np.min(epoch_mat_no_repeat, axis = 2)
    # zscores were across trials
    ranges_zscore = scipy.stats.zscore(ranges_each_trial, axis = 0)
    bad_trials = np.any(ranges_zscore > alpha, axis = 1)
    n_bad_trials[j] = bad_trials.sum()
    proportion_bad_trials[j] = bad_trials.mean()
    
    
    im_id1 = im_id_no_repeat[bad_trials == 0]
    epoch_mat1 = epoch_mat_no_repeat[bad_trials == 0]
    n_im = len(np.unique(im_id_no_repeat))
    ave_data = np.zeros([n_im, n_channels, n_times])
    for i in range(n_im):
        if np.sum(im_id1 == i) >0:
            ave_data[i] = np.mean(epoch_mat1[im_id1 == i,:,:], axis = 0)
        else:
            print subj, j
            print "im %d not found" %i   
            
    
print (n_bad_trials)

print (n_bad_trials.mean())
print (n_bad_trials.std())

print proportion_bad_trials
print (proportion_bad_trials.mean())
print (proportion_bad_trials.std())
    
    
   

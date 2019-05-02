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
import matplotlib.pyplot as plt
import scipy
import scipy.spatial

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
from HSIC import get_time_series_HSIC


#=======================================================================================
def get_sensor_HSIC(subj, ave_mat_path, X, ch_select = None, n_perm = 50,
                    deg = 100, perm_seq = None):
    """
    subj = "Subj2"
    ave_mat_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epochs_raw_data/"\
                 + "Subj2/%s_%s_ave.mat" %(subj, 1_50Hz_raw_ica_window_50ms)
    """
    mat_name = ave_mat_path
    mat_dict = scipy.io.loadmat(mat_name)
    ave_data = mat_dict['ave_data']
    picks_all = mat_dict['picks_all'][0]
    times = mat_dict['times'][0]
    n_im = mat_dict['n_im'][0][0]
    
    n_channel = len(picks_all)
    
    if ch_select is not None:
        picks = np.array([i  for i in range(n_channel) if picks_all[i] in ch_select ])
    else:
        picks = np.arange(0,n_channel)
    
    ave_data = ave_data[:, picks, :]
    # demean
    ave_data -= np.mean(ave_data, axis = 0)
    
    # Note: in Gretton's code, bandwidth for X and Y were determined by half of the distance between data samples
    # ave data needs to be normalized, no I need to normalized each time point, 
    # for RDM, this was not necessisary because it natrually had some normalization 
    # normalize such that the average distance between samples are equivalent between different time points. 
    ave_data_normalized = ave_data.copy()
    n_times = len(times)
    for t in range(n_times):
        tmp_scale = np.sum(np.abs(ave_data[:,:,t].max(axis = 0) - ave_data[:,:,t].min( axis = 0)))
        ave_data_normalized[:,:,t] = ave_data_normalized[:,:,t]/tmp_scale
    
    
    X -= np.mean(X)
    tmp_scale = np.sum(np.abs(X.max(axis = 0) - X.min(axis = 0)))
    X_normalized = X/tmp_scale
    
    orig_seq = np.arange(0,n_im)
    if perm_seq is None:
        perm_seq = np.zeros([n_perm,n_im])
        for i in range(n_perm):
            perm_seq[i] = orig_seq[np.random.permutation(orig_seq)]
            perm_seq = perm_seq.astype(np.int)
    
    result = get_time_series_HSIC(X_normalized,ave_data_normalized,perm_seq, deg = deg) 
    result['times'] = times
    return result
        


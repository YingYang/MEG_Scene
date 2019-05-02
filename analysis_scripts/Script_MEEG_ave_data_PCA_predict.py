# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import mne, time
mne.set_log_level('WARNING')
import numpy as np
import scipy.stats
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io

import sys
path0 = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/analyze_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)


meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

subj_list = range(1,9)
n_subj = len(subj_list)
relative_MSE_all = np.zeros(n_subj, dtype = np.object)
MSE_all = np.zeros(n_subj, dtype = np.object)


offset =0.03
# one Subject
for i in range(n_subj):
    
    subj = "Subj%d" %subj_list[i]
    percent = 100
    MEG_fname_suffix = "1_50Hz_raw_ica_window_50ms_first_%dpercent" %percent
    MEG_ave_mat_path = meg_dir + "%s/%s_%s_ave.mat" %(subj,subj,MEG_fname_suffix)
    data = scipy.io.loadmat(MEG_ave_mat_path)
    MEG_time_ms = ((data['times'][0]-offset)*1000).astype(np.int)
    MEG_data = data['ave_data']
    MEG_picks_all = data['picks_all'][0]
    del(data)
    
    EEG_fname_suffix = "1_50Hz_raw_ica_window_50ms"
    EEG_ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s_ave.mat" %(subj,subj,EEG_fname_suffix)
    data = scipy.io.loadmat(EEG_ave_mat_path)
    EEG_time_ms = (data['times'][0]*1000).astype(np.int)
    EEG_data = data['ave_data']
    EEG_picks_all = data['picks_all'][0]
    del(data)
    
    #==================================================================================
    # take the intersection of time points
    common_time  = np.intersect1d(MEG_time_ms, EEG_time_ms)
    MEG_in_common_time_id = [l for l in range(len(MEG_time_ms)) if MEG_time_ms[l] in common_time]
    EEG_in_common_time_id = [l for l in range(len(EEG_time_ms)) if EEG_time_ms[l] in common_time]
    MEG_data = MEG_data[:,:,MEG_in_common_time_id]
    EEG_data = EEG_data[:,:,EEG_in_common_time_id]
    
    # zscore everything, then normalize by number of sensors, so that the total 
    # variance of the two modalities are the same
    normalize_flag = "modality_normalize"
    #normalize_flag = "modality_normalize"
    if normalize_flag in ['sensor_normalize']:
        MEG_data = scipy.stats.zscore(MEG_data, axis = 0)
        EEG_data = scipy.stats.zscore(EEG_data, axis = 0)
    elif normalize_flag in ['modality_normalize']:
        MEG_data = scipy.stats.zscore(MEG_data, axis = 0)/np.sqrt(MEG_data.shape[1])
        EEG_data = scipy.stats.zscore(EEG_data, axis = 0)/np.sqrt(EEG_data.shape[1])
    
    MEG_ind = np.arange(0,306)
    EEG_ind = np.arange(306,306+128)
    data = np.concatenate([MEG_data, EEG_data], axis = 1)
    
    
    n_im = data.shape[0]
    n_train = 181
    train_im_set = np.random.choice(np.arange(n_im), n_train, replace = False )
    test_im_set = np.setdiff1d(np.arange(n_im), train_im_set)
    data_train = data[train_im_set]
    data_test = data[test_im_set]
    
    n_dim = 30
    cum_var_threshold = 0.8
    flag_use_var_thr = False
    n_times = len(common_time)
    
    relative_MSE = np.zeros([2,n_times])
    MSE = np.zeros([2, n_times])
    # for each time point, PCA and then predict
    for t in range(n_times):
        tmp_data = data[:,:,t]
        tmp_data_train = tmp_data[train_im_set]
        tmp_data_test = tmp_data[test_im_set]
    
        u,d,v = np.linalg.svd(tmp_data_train)
        cum_persent = np.cumsum(d**2/np.sum(d**2))
        if flag_use_var_thr:
            tmp_dim = np.nonzero(cum_persent>= cum_var_threshold)[0][0]+1
        else:
            tmp_dim = n_dim
        MEG_proj = v[0:tmp_dim, MEG_ind]
        EEG_proj = v[0:tmp_dim, EEG_ind]
        
        # use EEG to predict MEG
        # data1 = udv1 
        # data2 = udv2
        # data2 = data1 v1.T v2
        MEG_pred = tmp_data_test[:,EEG_ind].dot(EEG_proj.T).dot(MEG_proj)
        MSE_MEG = np.mean((tmp_data_test[:, MEG_ind]-MEG_pred)**2)
        relative_MSE_MEG = MSE_MEG/ np. mean(tmp_data_test[:, MEG_ind]**2)
        
        EEG_pred = tmp_data_test[:,MEG_ind].dot(MEG_proj.T).dot(EEG_proj)
        MSE_EEG = np.mean((tmp_data_test[:, EEG_ind]-EEG_pred)**2)
        relative_MSE_EEG = MSE_EEG/ np. mean(tmp_data_test[:, EEG_ind]**2)
        relative_MSE[0,t] = relative_MSE_MEG
        relative_MSE[1,t] = relative_MSE_EEG
        
        MSE[0,t] = MSE_MEG
        MSE[1,t] = MSE_EEG
        
    
    relative_MSE_all[i] = relative_MSE
    MSE_all[i] = MSE

relative_MSE_mat = np.zeros([n_subj, 2, n_times])
for i in range(n_subj):
    relative_MSE_mat[i] = relative_MSE_all[i]
    
MSE_mat = np.zeros([n_subj, 2, n_times])
for i in range(n_subj):
    MSE_mat[i] = MSE_all[i]    
    
plt.figure()
plt.plot(common_time, relative_MSE_mat.mean(axis = 0).T, lw = 2)
plt.grid()  
plt.legend(['EEG predicting MEG', 'MEG predicting EEG'])
    
plt.figure()
for i in range(2):
    plt.subplot(2,1,i+1)
    plt.imshow(relative_MSE_mat[:,i,:], interpolation = "none")
    plt.colorbar()

plt.figure()
plt.plot(common_time, MSE_mat.mean(axis = 0).T, lw = 2)
plt.grid()  
plt.legend(['EEG predicting MEG', 'MEG predicting EEG'])

plt.figure()
for i in range(2):
    plt.subplot(2,1,i+1)
    plt.imshow(MSE_mat[:,i,:], interpolation = "none")
    plt.colorbar()
# -*- coding: utf-8 -*-
"""
Get source space solution for different lambda

@author: ying
"""

import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import scipy.io
import scipy.spatial
from copy import deepcopy
import sys
#path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path1 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
#path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)


data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
data_root_dir =data_root_dir0 +"MEG_preprocessed_data/MEG_preprocessed_data/"



MEGorEEG = ['EEG','MEG']

#for isMEG in [0,1]:

# redo the EEG source space solutions

flag_swap_PPO10_POO10 = True
#if True:
    #isMEG = True
lambda2_seq = [0.1, 1, 10]

for lambda2 in lambda2_seq:
    isMEG = True
    if isMEG:
        subj_list = range(1, 19)
    
    else:
        subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
        
    
    MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect" \
        if flag_swap_PPO10_POO10 else "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"    
    fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
    
    n_subj = len(subj_list)

    print ("isMEG %d" %isMEG)
    print (fname_suffix)
    print ("lambda = %1.2f" %lambda2)
    stc_out_dir =  data_root_dir0 + "Result_Mat/" \
                + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]
    n_im = 362
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        
        if isMEG:
            fwd_path = data_root_dir + "/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
            epochs_path = (data_root_dir \
                         +"/epoch_raw_data/%s/%s_%s" \
                        %(subj, subj, "run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"))
            datapath = data_root_dir + "/epoch_raw_data/%s/%s_%s.mat" \
                      %(subj, subj, fname_suffix)
        else:
            # no full EEG data yet, to be added later
            fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                       + "EEG_DATA/DATA/fwd/%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)
            if flag_swap_PPO10_POO10:
                epochs_path = ("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                         +"EEG_DATA/DATA/epoch_raw_data/%s_EEG/%s_EEG_%s" \
                        %(subj, subj, "filter_1_110Hz_notch_PPO10POO10_swapped_ica_reref_smoothed-epo.fif.gz"))
            else:
                epochs_path = ("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                         +"EEG_DATA/DATA/epoch_raw_data/%s_EEG/%s_EEG_%s" \
                        %(subj, subj, "filter_1_110Hz_notch_ica_reref_smoothed-epo.fif.gz"))
            datapath = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                      + "EEG_DATA/DATA/epoch_raw_data/%s_EEG/%s_EEG_%s.mat" \
                      %(subj, subj, fname_suffix)
                      
        mat_data = scipy.io.loadmat(datapath)
        data = mat_data['ave_data']            
        # load data
        epochs = mne.read_epochs(epochs_path)
        if isMEG:
            epochs1 = mne.epochs.concatenate_epochs([epochs, deepcopy(epochs)])
            epochs = epochs1[0:n_im]
            epochs._data = data.copy()
            del(epochs1)
        else:
            epochs = epochs[0:n_im]
            epochs._data = data.copy()

        # a temporary comvariance matrix
        cov = mne.compute_covariance(epochs, tmin=None, tmax=-0.05)
        # load the forward solution
        # mne newer version change
        #fwd = mne.read_forward_solution(fwd_path, surf_ori = True)
        fwd = mne.read_forward_solution(fwd_path)
        fwd = mne.convert_forward_solution(fwd, surf_ori = True,)
        # create inverse solution
        inv_op = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, depth = 0.8,
                                                        fixed = True)
        stc = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2 = lambda2,
                                                    method = "dSPM")
        
        source_data = np.zeros([n_im, stc[0].data.shape[0], stc[0].data.shape[1]])
        for j in range(n_im):
            source_data[j] = stc[j].data 
            
        times = epochs.times
        times_ind = times>=-0.1
        times = times[times_ind]
        source_data = source_data[:,:, times_ind]
        del(stc)

        mat_name = stc_out_dir + "%s_%s_%s_lambda2_%1.1f_ave.mat" %(
                subj, MEGorEEG[isMEG],fname_suffix, lambda2)
        # no offset is applied here
        time_corrected = 0
        mat_dict = dict(source_data = source_data, times = times,
                        time_corrected = time_corrected)
        scipy.io.savemat(mat_name, mat_dict)
        
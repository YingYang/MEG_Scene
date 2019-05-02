# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import numpy as np
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)




meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
MEGorEEG = ["EEG","MEG"] 

Subj_list = range(1,19)
n_Subj = len(Subj_list)
Subj_has_EEG = np.ones([n_Subj], dtype = np.bool)
Subj_has_EEG[[8,14,16]] = False


common_times = np.round(np.arange(-0.1, 0.8, 0.01), decimals = 2) 
n_times = len(common_times)
MEG_offset = 0.04


n_im = 362
mask = np.ones([n_im, n_im])
mask = np.triu(mask,1)
n_pair = np.int(mask.sum())

rsm_all = np.zeros([2,n_Subj, n_pair, n_times])

for isMEG in [0,1]:
    for j in range(n_Subj):
        subj = "Subj%d" %Subj_list[j]
        print subj, MEGorEEG[isMEG]
        if isMEG:
            ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
        else:
            if Subj_has_EEG[j]:
                ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
            else:
                continue
        ave_mat = scipy.io.loadmat(ave_mat_path)
        offset = MEG_offset if isMEG else 0.0
        times = np.round(ave_mat['times'][0] - offset, decimals = 2)
        time_ind = np.all(np.vstack([times <= common_times[-1], times >= common_times[0]]), axis = 0)
        ave_data = ave_mat['ave_data'][:, :, time_ind]
        ave_data -= ave_data.mean(axis = 0) 
        
        for t in range(n_times):
            tmp = ave_data[:,:,t]
            corr = np.corrcoef(tmp, rowvar = 1)
            rsm_all[isMEG,j,:,t] = corr[mask>0]
   
# compute the leave one out correlation
corr_rsm_loo = np.zeros([2, n_Subj, n_times])
for isMEG in [0,1]:
    for j in range(n_Subj):  
        if isMEG:
            valid_subj = range(n_Subj)  
        elif Subj_has_EEG[j]:
            valid_subj = np.nonzero(Subj_has_EEG)[0]
        else:
            continue
        other_subj = np.setdiff1d(valid_subj, j)
        tmp1 = rsm_all[isMEG, j,:,:]
        tmp2 = rsm_all[isMEG, other_subj, :,:].mean(axis = 0)
        for t in range(n_times):
            corr_rsm_loo[isMEG, j, t] = np.corrcoef(tmp1[:,t],tmp2[:,t])[0,1]


fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_rsm/"            
vmin, vmax = -0.2, 1.0
plt.figure()
for isMEG in [0,1]:
    plt.subplot(2,1,isMEG+1);
    plt.imshow(corr_rsm_loo[isMEG], aspect = "auto", interpolation = "none",
               vmin = vmin, vmax = vmax);
    plt.colorbar(); plt.title(MEGorEEG[isMEG])
plt.savefig(fig_outdir+ "corr_rsm_loo_individual.pdf")

    
plt.figure()
for isMEG in [0,1]:
    if isMEG:
        plt.plot(common_times, corr_rsm_loo[isMEG].mean(axis = 0));
    else:
        plt.plot(common_times, corr_rsm_loo[isMEG, Subj_has_EEG>0].mean(axis = 0));
plt.legend(MEGorEEG); plt.xlabel('time (s)'); plt.ylabel('loo rsm corr');
plt.grid('on')
plt.savefig(fig_outdir+ "corr_rsm_loo_mean.pdf")        
        
            
    

    
    
    
    
 

    
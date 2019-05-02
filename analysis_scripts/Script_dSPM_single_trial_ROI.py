# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

# For Jordan only

# 
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
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
subj = "Subj10"



ROI_bihemi_names = [ 'pericalcarine',  #'medialorbitofrontal',
                        'PPA_c_g'] #'TOS_c_g', 'RSC_c_g']
nROI = len(ROI_bihemi_names)                    
labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"  
   
    
times = np.arange(0.01,0.96,0.01)
n_times = 95

labeldir1 = labeldir + "%s/" % subj
labels_bihemi = list()
for j in ROI_bihemi_names:
    tmp_label_list = list()
    for hemi in ['lh','rh']:
        print subj, j, hemi
        tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,j,hemi)
        tmp_label = mne.read_label(tmp_label_path)
        tmp_label_list.append(tmp_label)
    labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1]) 


   
fname_suffix = "1_110Hz_notch_ica_all_trials"
fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
           + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
epochs_path = ("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
             +"MEG_DATA/DATA/epoch_raw_data/%s/%s_%s" \
            %(subj, subj, "run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"))
datapath = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
          + "MEG_DATA/DATA/epoch_raw_data/%s/%s_%s.mat" \
          %(subj, subj, fname_suffix)

mat_data = scipy.io.loadmat(datapath)
data = mat_data['epoch_mat_no_repeat'] 
im_id = mat_data['im_id_no_repeat']           
# load data
epochs = mne.read_epochs(epochs_path)

n_trial = data.shape[0]
epochs1 = deepcopy(epochs)
for l in range(10):
    epochs1 = mne.epochs.concatenate_epochs([epochs1, deepcopy(epochs)])

epochs1 = epochs1[0:n_trial]
epochs = epochs1.copy()
del(epochs1)
# a temporary comvariance matrix
cov = mne.compute_covariance(epochs, tmin=None, tmax=-0.05)
# load the forward solution
fwd = mne.read_forward_solution(fwd_path, surf_ori = True)
# create inverse solution
inv_op = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, depth = 0.8,
                                                fixed = True)

ROI_data = list()
n_ROI = len(labels_bihemi)
for i in range(2):                                          
    stc = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2 = 1.0, method = "dSPM",
                                                label = labels_bihemi[i])
    source_data = np.zeros([n_trial, stc[0].data.shape[0], stc[0].data.shape[1]])
    for j in range(n_trial):
        source_data[j] = stc[j].data 
            
    times = epochs.times
    times_ind = times>=0.05
    times = times[times_ind]
    source_data = source_data[:,:, times_ind]
    del(stc)
    ROI_data.append(source_data)

#time_corrected = 0
mat_name = "/home/ying/Dropbox/tmp/NEIL_MEG_One_Subj_V1_PPA_dSPM_single_trials.mat"
mat_dict = dict(ROI_names = ROI_bihemi_names, ROI_data1 = ROI_data[0], ROI_data2 = ROI_data[1],
                times = times, subj = subj, im_id = im_id)
scipy.io.savemat(mat_name, mat_dict)
        
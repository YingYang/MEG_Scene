# -*- coding: utf-8 -*-
"""
Somehow, this did not work well? The permuted values were so close to each other? Bug somwhere?

@author: ying
"""

import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)

from sensor_space_HSIC import get_sensor_HSIC 


isMEG = True
Subj_list = range(1,10)
n_Subj = len(Subj_list)
offset = 0.04
n_im = 362
n_perm = 50
orig_seq = np.arange(0,n_im)
perm_seq = np.zeros([n_perm, n_im])
for k in range(n_perm):
    perm_seq[k] = np.random.permutation(orig_seq)
    perm_seq = perm_seq.astype(np.int)

meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
MEGorEEG = ['EEG','MEG']
    #
fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    
mat_file_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_dependence/hsic/"



model_name = "AlexNet"
feature_suffix = "no_aspect"
feat_name_seq = ["rawpixelgraybox"]
if True:
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]
    #feat_name_seq = list()
    for l in layers:
        feat_name_seq.append("%s_%s_%s" %(model_name,l, feature_suffix))
n_feat_name = len(feat_name_seq) 
regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
offset = 0.04 if isMEG else 0
        
   
feat_name = "fc6_no_aspect"     
mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/AlexNet_Features/AlexNet_%s.mat" %feat_name)
data = mat_data['data']
X0 = data - np.mean(data, axis = 0)

  
# for each feature, always demean them first
X = X0- np.mean(X0, axis = 0)

# note, the deg (bandwidwith) for the kernel is really important. Gretton's code uses half of the distance as the bandwidth.  
#for i in range(n_Subj):
if False:
    i = 0
    subj = "Subj%d" %Subj_list[i]
    if isMEG:
        ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
    else:
        ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)

    result_hsic = get_sensor_HSIC(subj, ave_mat_path, X, ch_select = None, n_perm = n_perm,
                        deg = 0.9, perm_seq = perm_seq) 
    plt.figure()
    plt.plot(result_hsic['p_time'])
    plt.plot(result_hsic['Stat_time'])
    plt.plot(result_hsic['Stat_perm_time'].T, 'r', alpha = 0.3)

    
if False:
    X_id = 0
    feat_name = "neil_attr"
    n_times = 110
    log10p_val_all = np.zeros([n_Subj,n_times]) 
    for i in range(n_Subj):
        subj = "Subj%d" %Subj_list[i]
        if isMEG:
            mat_name = mat_file_out_dir + "%s_MEG_result_hsic_%s.mat" %(subj, feat_name)
        else:
            mat_name = mat_file_out_dir + "%s_EEG_result_hsic_%s.mat" %(subj, feat_name)
            
        tmp_mat = scipy.io.loadmat(mat_name)
        log10p_val_all[i] = -np.log10(tmp_mat['p_time'][0])
        times = tmp_mat['times'][0]
 
    offset = 0.04
    times_in_ms = (times - offset )*1000.0      
    
    plt.figure()
    plt.imshow(log10p_val_all, aspect = "auto", interpolation = "none",
               extent = [times_in_ms[0], times_in_ms[-1], 0, n_Subj], origin = "lower")
    plt.colorbar()
    plt.xlabel("time (ms)" )
    plt.ylabel("subject id")
                
            
    # Fisher's method    
    from Stat_Utility import Fisher_method
    log10p_fisher = -np.log10(Fisher_method(10**(-log10p_val_all), axis = 0))
    
    plt.figure()
    plt.plot(times_in_ms, log10p_fisher)
    plt.grid("on")
    plt.xlabel( "time (ms)")
    plt.ylabel( "-log10 p fisher")
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/HSIC/"
    plt.savefig(fig_outdir + "Subj1_9_Fisher_HSIC_%s.pdf" %feat_name)
                
            
        
    
    

# -*- coding: utf-8 -*-
"""
Additional script for sensor space OLS regression of the additional 3 subjects where presentation was 500 ms
@author: ying
"""

import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import time
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
from ols_regression import ols_regression
from Stat_Utility import bootstrap_mean_array_across_subjects

from Script_sensor_ols_regression import sensor_space_regression_no_regularization

#=======================================================================================
if __name__ == "__main__":
    meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
    eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
    MEGorEEG = ['EEG','MEG']
    # try unsmoothed
    flag_swap_PPO10_POO10 = True
    MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect" \
        if flag_swap_PPO10_POO10 else "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"    
    
    model_name = "AlexNet"
    
    isMEG = 0
    #Subj_list = ['SubjYY_100', 'SubjYY_200', 'SubjYY_500'] 
    Subj_list = ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]
    n_times = 109
    n_channels = 128
    
    Flag_CCA = True
    n_Subj = len(Subj_list)
    n_im = 362
    fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
    
        
    regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
    mat_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/ave_ols/"
    sep_CCA = True
               # CCA determined by additional training data, only the common 10 dimensions of the first component is regressed out
    if sep_CCA == True:
        n_comp1 = 6
        n_comp = 6
        feature_suffix = "no_aspect"
        #feat_name_seq = ['conv1','fc6']
        feat_name_seq = ['conv1_nc160','fc7_nc160']
        mode_id = 3
        mode_names = ['','Layer1_6','localcontrast_Layer6', 'Layer1_7_noncontrast160']
    
        X_list = list()
        for j in range(2):
            #tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_cca_%d_residual_svd.mat" %(feat_name_seq[j], n_comp1))
            mat_name = regressor_dir+ "AlexNet_%s_%s_cca_%d_residual_svd.mat" %(mode_names[mode_id], feat_name_seq[j], n_comp1)
            tmp = scipy.io.loadmat(mat_name)
            X_list.append(tmp['u'][:,0:n_comp])
        
        X_list.append(tmp['merged_u'][:,0:n_comp])
        feat_name_seq.append( "CCA_merged")
    
    # add another set, the top n_compnent 
    fname = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/Stim_images_Layer1_contrast_noaspect.mat"
    mat_dict = scipy.io.loadmat(fname)
    tmp = mat_dict['contrast_noaspect']
    tmp = tmp- np.mean(tmp, axis = 0)
    U,D,V = np.linalg.svd(tmp)
    var_prop = np.cumsum(D**2)/np.sum(D**2)
    
    X_list.append(U[:,0:n_comp])
    feat_name_seq.append("contrast")
    n_feat = len(X_list)
    
    log10p = np.zeros([n_Subj, n_feat, n_channels, n_times])
    Rsq = np.zeros([n_Subj, n_feat, n_channels, n_times])
    for j in range(n_feat):
        X = X_list[j]
        for i in range(n_Subj):
            t0 = time.time()
            subj =Subj_list[i]
            #subj = Subj_list[i]
            if isMEG:
                ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
            else:
                ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
            result_reg = sensor_space_regression_no_regularization(subj, ave_mat_path, X)
            log10p[i,j] = result_reg['log10p'][:,0:n_times]
            Rsq[i,j] = result_reg['Rsq'][:,0:n_times]
            times = result_reg['times'][0:n_times]
            
            n_channel = result_reg['log10p'].shape[0]
     
     
    # save the results as mat
    mat_name = mat_out_dir + "AlexNet_%s_CCA%d_ncomp%d_ave_%s_additional_subjs.mat" %(MEGorEEG[isMEG], sep_CCA, n_comp, fname_suffix)
    print mat_name
    mat_dict = dict(Rsq = Rsq, times = times, log10p = log10p, Subj_list = Subj_list,
                    isMEG = isMEG, X_list = X_list, n_comp = n_comp, 
                    feat_name_seq = feat_name_seq, n_feat = n_feat)    
    scipy.io.savemat(mat_name, mat_dict)

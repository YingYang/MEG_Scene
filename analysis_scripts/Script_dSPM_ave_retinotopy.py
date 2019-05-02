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

MEGorEEG = ['EEG','MEG']
if True:
    isMEG = 1
    subj_list = range(1, 14)
    
if False:
    isMEG= 0
    subj_list = [4]
    
n_subj = len(subj_list)
        
    
    
offset = 0.04
fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"
n_im = 362
    
model_name = "AlexNet"
names = ["meridian2","ecc2", "lr2", "ul2"]
d = 5 

n_set = len(names)
for l in range(n_set):
    mat_outname ="/home/ying/Dropbox/Scene_MEG_EEG/Features/" \
                        + "Retinotopy/%s_conv1_%s.mat" % (model_name,names[l])
    data = scipy.io.loadmat(mat_outname)['data']
    n_group = data.shape[1]
    # just do first PC
    X = data[:,:,0:d].reshape([n_im, n_group*d]) 
    X_aug = np.zeros([n_im, X.shape[1]+1])
    X_aug[:,0] = 1.0
    X_aug[:,1::] = X.copy() 
    
    invXXT = np.linalg.inv(X_aug.T.dot(X_aug))                  
    
    stc_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/" \
                + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        print subj
        mat_name = stc_out_dir + "%s_%s_%s_ave.mat" %(subj, MEGorEEG[isMEG],fname_suffix)
        # no offset is applied here
        mat_dict = scipy.io.loadmat(mat_name)
        source_data = mat_dict['source_data']
        times = mat_dict['times'][0]
        time_corrected = mat_dict['time_corrected'][0][0]
        
        if time_corrected == 0:
            times -= offset
            
        # do regression
        n_dipoles = source_data.shape[1]
        n_times = source_data.shape[2]
        # mean difference between the groups
        mean_contrast = np.zeros([n_dipoles, n_times])
        se_error = np.zeros([n_dipoles, n_times]) 
        T_contrast = np.zeros([n_dipoles, n_times])
        
        
        for j in range(n_dipoles):
            for t in range(n_times):
                tmp = source_data[:,j,t]
                beta = invXXT.dot( X_aug.T.dot(tmp))
                beta_valid = np.reshape(beta[1::],[n_group, d])
                mean_contrast[j,t] = np.mean(beta_valid[0]**2-beta_valid[1]**2)
                res = tmp - X_aug.dot(beta)
                se_error[j,t] = np.sqrt(np.sum(res**2)/(n_im - X_aug.shape[1]))
                
                if d == 1:
                    weight = np.array([0,1,-1])
                    T_contrast[j,t] = np.dot(weight, beta)/np.sqrt((weight.dot(invXXT)).dot(weight))/se_error[j,t]
        
        if d>1:
            # this is no longer T-statistics
            T_contrast = mean_contrast/se_error**2
                
   
        #================================================================================
        # save results in stc
        if isMEG:
            fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                       + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
        else:
            fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                       + "EEG_DATA/DATA/fwd/%s/%s_EEG-fwd.fif" %(subj, subj)
        fwd = mne.read_forward_solution(fwd_path)
        src = fwd['src']
        vertices = [ src[0]['vertno'], src[1]['vertno']]
        stc_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/mne_ols_ave_data_%s/" %(MEGorEEG[isMEG])
        stc = mne.SourceEstimate(data = T_contrast,vertices = vertices, tmin = times[0], tstep = times[2]-times[1] )
        stc_name = stc_path   + "%s/%s_reti_%s_dim%d_mne_ols_ave_data_%s" %( subj, subj, names[l], d, MEGorEEG[isMEG])
        stc.save(stc_name)
        del(stc)
        
        if False:
            tmin, tmax = 0.06, 0.12
            tind = np.all(np.vstack([times>= tmin, times<=tmax]), axis = 0)
            T_contrast_mean = T_contrast[:,tind].mean(axis = 1)
            stc = mne.SourceEstimate(data = T_contrast_mean[:, np.newaxis],
                                     vertices = vertices, tmin = times[0], tstep = times[2]-times[1] )
            stc_name = stc_path   + "%s/%s_reti_%s_dim%d_mne_ols_ave_data_%s_tave" %( subj, subj, names[l], d, MEGorEEG[isMEG])
            stc.save(stc_name)
            del(stc)
            
            
       
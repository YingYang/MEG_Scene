# -*- coding: utf-8 -*-
"""
Not used. such regression were done within ROIs or on the morphed data. 
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
from ols_regression import ols_regression

flag_swap_PPO10_POO10 = True
MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect" \
   if flag_swap_PPO10_POO10 else "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"


MEGorEEG = ['EEG','MEG']
for isMEG in [False, True]:
    if isMEG:
        subj_list = range(1, 19)
    else:
        subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
    
    fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
    n_subj = len(subj_list)
          
    offset = 0.04 if isMEG else 0.00
    regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
    n_im = 362
    if False:
        n_dim = 15
        model_name = "AlexNet"
        layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]
        feat_name_seq = list()
        for l in layers:
            feat_name_seq.append("%s_%s" %(model_name,l))
        n_feat_name = len(feat_name_seq) 
        X_list = list()
        for j in range(n_feat_name):  
            # load the design matrix 
            feat_name = feat_name_seq[j]
            regressor_fname = regressor_dir +  "%s_PCA.mat" %(feat_name)
            tmp = scipy.io.loadmat(regressor_fname)
            X0 = tmp['X'] 
            X = X0[:,0:n_dim]
            X -= np.mean(X, axis = 0)
            X_list.append(X)
        
    if False:
        dim = 15
        n_dim = dim
        fname = "AlexNet_conv1_%d_AlexNet_fc6_%d_intersect_%d" %(dim, dim, dim)
        regressor_mat = scipy.io.loadmat(regressor_dir + fname + ".mat")
        sub_feat_seq = ['Xoverlap','XresA','XresB']
        n_sub_feat = len(sub_feat_seq)
        X_list = list()
        for j in range(n_sub_feat):
            X_list.append(regressor_mat[sub_feat_seq[j]])
        feat_name_seq = ['AlexNet_conv1_fc6_common', 'AlexNet_conv1_res','AlexNet_fc6_res']
        n_feat_name = len(feat_name_seq)     
        
        
    m1,m2 = 4,2
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_reg/"
    stc_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/" \
                + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
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

        F_val = np.zeros([n_feat_name, n_dipoles, n_times])
        log10p = np.zeros([n_feat_name, n_dipoles, n_times])
       
        for j in range(n_feat_name):  
            X = X_list[j]
            tmp_result = ols_regression(source_data, X)
            F_val[j] = tmp_result['F_val']
            log10p[j] = tmp_result['log10p']
            
        plt.figure(figsize = (12,12))
        for j in range(n_feat_name):
            plt.subplot(m1,m2,1+j)
            plt.imshow(log10p[j], aspect = "auto", interpolation = "none", 
                       extent = [times[0]*1000, times[-1]*1000, 1, n_dipoles],
                       origin = "lower")
            plt.colorbar()
            plt.title(feat_name_seq[j])
            plt.xlabel('time ms')
            plt.ylabel( "dipole id")
        fig_path = fig_outdir + "%s_%s_source_reg_%dpc.eps" % (subj, MEGorEEG[isMEG], n_dim)
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
   
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
        stc_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/dSPM_ols_ave_data_%s/" %(MEGorEEG[isMEG])
        for j in range(n_feat_name):
            stc_log10p = mne.SourceEstimate(data = log10p[j],vertices = vertices, tmin = times[0], tstep = times[2]-times[1] )
            stc_name = stc_path   + "%s/%s_%s_dim%d_mne_ols_ave_data_%s_%s" %( subj, subj, feat_name_seq[j], n_dim, MEGorEEG[isMEG], fname_suffix)
            stc_log10p.save(stc_name)
           

if False:
    feat_name_seq = ["conv1","conv2","conv5","fc6", "fc7", "neil_attr", "sun_hierarchy"]
    n_feat_name = len(feat_name_seq)
    subjects_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
    stc_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/dSPM_ols_ave_data_%s/" %(MEGorEEG[isMEG])
    fig_path =  "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_reg/surface_view_%s/" %(MEGorEEG[isMEG])
    clim = dict(kind='value', lims=[2.0, 7.0, 15.0])
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        for j in range(n_feat_name):
            message = 'mne_ave_%s_%s_%dpc_log10p' %( subj, feat_name_seq[j], n_dim)
            stc_name = stc_path   + "%s/%s_%s_dim%d_mne_ols_ave_data_%s_%s" %( subj, subj, feat_name_seq[j], n_dim, MEGorEEG[isMEG], fname_suffix)
            stc = mne.read_source_estimate(stc_name)
            brain = stc.plot(surface='inflated', hemi='both', subjects_dir=subjects_dir,
                        subject = subj,  clim=clim)
            for k in range(0,90,5):
                brain.set_data_time_index(k)
                for view in ['ventral','dorsal']:
                    brain.show_view(view)
                    im_name = fig_path + "%s/mne_ave_%s_%s_%dpc_log10p_%03dms_%s_%s.png" \
                       %(subj, subj, feat_name_seq[j], n_dim, np.int(np.round(stc.times[k]*1000)), view, MEGorEEG[isMEG])
                    brain.save_image(im_name)          
            brain.close()
            
            for hemi in ['lh','rh']:
                brain = stc.plot(surface='inflated', hemi= hemi, subjects_dir=subjects_dir,
                        subject = subj,  clim=clim)
                for k in range(0,90,5):
                    brain.set_data_time_index(k)
                    for view in ['lateral','medial']:
                        brain.show_view(view)
                        im_name = fig_path + "%s/mne_ave_%s_%s_%dpc_log10p_%03dms_%s_%s_%s.png" \
                           %(subj, subj, feat_name_seq[j], n_dim, np.int(np.round(stc.times[k]*1000)), hemi, view, MEGorEEG[isMEG])
                        brain.save_image(im_name)          
                brain.close()
            

    
    
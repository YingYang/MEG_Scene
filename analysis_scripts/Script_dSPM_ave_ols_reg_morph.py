# -*- coding: utf-8 -*-
"""
"""

import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import scipy.io
import scipy.spatial
from copy import deepcopy
import sys
#path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/"

sys.path.insert(0, path0)
#path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
path1 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"

sys.path.insert(0, path1)
import os
from ols_regression import ols_regression

#%%
data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
data_root_dir =data_root_dir0 +"MEG_preprocessed_data/MEG_preprocessed_data/"



#=== load the source space of the fsaverage subject
subjects_dir = data_root_dir0 + "/FREESURFER_ANAT/"

#data_path = mne.datasets.sample.data_path()
data_path = data_root_dir0+"MNE-sample-data/"
subjects_dir_mne = os.path.join(data_path, 'subjects')
fname_src_fs = os.path.join(subjects_dir_mne, 'fsaverage', 'bem',
                       'fsaverage-ico-5-src.fif')
src_to = mne.read_source_spaces(fname_src_fs)                      
subject_to = "fsaverage" 
#this is also 10242*2
vertices_to = [src_to[0]['vertno'], 
               src_to[1]['vertno']]
#vertices_to = [np.arange(10242), np.arange(10242)]
n_vertices =  src_to[0]['nuse']+  src_to[1]['nuse']                    


#isMEG = False
isMEG = True
MEGorEEG = ['EEG','MEG']


print MEGorEEG[isMEG]

flag_swap_PPO10_POO10 = True
MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect" \
    if flag_swap_PPO10_POO10 else "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"    
fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix


if isMEG:
    subj_list = range(1, 19)
    times = np.arange(0.01,1.01,0.01)
    n_times = 100
    time_in_ms = (times - 0.04)*1000.0    
else:
    subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
    times = np.arange(0.01,1.0,0.01)
    n_times = 99
    time_in_ms = times*1000.0
    
  
n_subj = len(subj_list)  
stc_out_dir = data_root_dir0 + "Result_Mat/" \
                + "source_solution/dSPM_%s_ave_per_im_morphed/" % MEGorEEG[isMEG]
n_im = 362
#%%    
if False:
#if True:
    #for i in range(n_subj):
    for i in range(10,18):
        subj = "Subj%d" %subj_list[i]
        '''
        if isMEG:
            fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                       + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
            epochs_path = ("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                         +"MEG_DATA/DATA/epoch_raw_data/%s/%s_%s" \
                        %(subj, subj, "run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"))
            datapath = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                      + "MEG_DATA/DATA/epoch_raw_data/%s/%s_%s.mat" \
                      %(subj, subj, fname_suffix)
        else:
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
        '''
        if isMEG:
            fwd_path = data_root_dir + "/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
            epochs_path = (data_root_dir \
                         +"/epoch_raw_data/%s/%s_%s" \
                        %(subj, subj, "run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"))
            datapath = data_root_dir + "/epoch_raw_data/%s/%s_%s.mat" \
                      %(subj, subj, fname_suffix)
        
        mat_data = scipy.io.loadmat(datapath)
        data = mat_data['ave_data']            
        # load data
        epochs = mne.read_epochs(epochs_path)
        epochs.resample(sfreq = 100.0)
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
        fwd = mne.read_forward_solution(fwd_path)
        fwd = mne.convert_forward_solution(fwd, surf_ori = True,)
        # create inverse solution
        inv_op = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, fixed = True, depth = 0.8)
        stc = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2 = 1.0, method = "dSPM")
        
        # compute the morphing matrix
        subject_from = subj
        # load src
        # this is too many indices, can we reduce them?
        morph_mat = mne.compute_morph_matrix(subject_from, subject_to,
                                     stc[0].vertices, vertices_to,
                                     subjects_dir=subjects_dir)                          
        source_data = np.zeros([n_im, n_vertices , stc[0].data.shape[1]])
        for j in range(n_im):
            tmp_stc_to = mne.morph_data_precomputed(subject_from, subject_to,
                                      stc[j], vertices_to, morph_mat)
            source_data[j] = tmp_stc_to.data 
            
        times = epochs.times
        times_ind_start = np.nonzero(times>=0.0)[0][0]
        times = times[times_ind_start::]
        del(stc)
        
        
        npy_name = stc_out_dir + "%s_%s_%s_ave_morphed.npy" %(subj, MEGorEEG[isMEG],fname_suffix)
        np.save(npy_name, source_data[:,:, times_ind_start::])
        # no offset is applied here     
    # just save times in to a mat file
    times_mat_name = stc_out_dir + "%s_ave_morphed_times.mat" %(MEGorEEG[isMEG])
    scipy.io.savemat(times_mat_name, dict(times =times, time_corrected = 0))

#%%        
# Regression
    
flag_cca =False

# assume for EEG there is 0.02 offset
offset = 0.04 if isMEG else 0.02

m1,m2 = 4,2
#regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
regressor_dir = data_root_dir0 + "/regressor/"

fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_reg/"
mat_out_dir = data_root_dir0 + "/Result_Mat/" \
            + "source_regression/dSPM_reg_%s_ave_perm_im_morph/" %MEGorEEG[isMEG]          
times_mat_name = stc_out_dir + "%s_ave_morphed_times.mat" %(MEGorEEG[isMEG])
mat_dict = scipy.io.loadmat(times_mat_name)
times = mat_dict['times'][0]
time_corrected = mat_dict['time_corrected'][0][0]
if time_corrected == 0:
    times -= offset
n_times = len(times)

if not flag_cca:  


    #feature_suffix = "no_aspect_no_contrast100"
    feature_suffix = "no_aspect_no_contrast160_all_im"
    n_dim = 10
    model_name = "AlexNet"
    #feat_name_seq = ["rawpixelgraybox"]
    feat_name_seq = []
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]
    #feat_name_seq = list()
    for l in layers:
        feat_name_seq.append("%s_%s_%s" %(model_name,l,feature_suffix))
    n_feat = len(feat_name_seq) 
    X_list = []
    for j in range(n_feat):  
        # load the design matrix 
        feat_name = feat_name_seq[j]
        print feat_name
        regressor_fname = regressor_dir +  "%s_PCA.mat" %(feat_name)
        tmp = scipy.io.loadmat(regressor_fname)
        X0 = tmp['X']
        X = X0[0:n_im,0:n_dim]
        X_list.append(X)

    
else:    
    # use the 6 cca components
    n_comp1 = 6
    n_comp = 6
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
    
n_feat = len(X_list) 
    
    

"""
n_comp = 20
model_name = "placeCNN"
fname = "%s_conv1_%d_%s_fc6_%d_intersect_%d.mat" \
        %(model_name, n_comp, model_name, n_comp, n_comp)
regressor_mat = scipy.io.loadmat(regressor_dir + fname)
sub_feat_seq = ['Xoverlap','XresA','XresB']
n_sub_feat = len(sub_feat_seq)
X_list = list()
for j in range(n_sub_feat):
    X = regressor_mat[sub_feat_seq[j]]
    X_list.append(X-np.mean(X, axis = 0))

feat_name_seq = ['%s_Layer1_6_overlap_%d' % (model_name,n_comp), 
                 '%s_Layer1_%d' % (model_name, n_comp),
                 '%s_Layer6_%d' % (model_name, n_comp)]
n_feat_name = len(feat_name_seq)
n_dim = n_comp
"""
 
# run the regression               
if True:    
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i] 
        npy_name = stc_out_dir + "%s_%s_%s_ave_morphed.npy" %(subj, MEGorEEG[isMEG],fname_suffix)
        source_data = np.load(npy_name)    
        # do regression
        n_dipoles = source_data.shape[1]
        for j in range(n_feat): 
            feat_name = feat_name_seq[j]
            print subj
            X = X_list[j]
            tmp_result = ols_regression(source_data, X)
            mat_name = mat_out_dir + "%s_%s_dim%d_dSPM_ols_ave_data_%s_morphed_%s.mat" \
                      %(subj, feat_name, n_dim, MEGorEEG[isMEG],fname_suffix)
            print mat_name
            mat_dict = dict(Rsq = tmp_result['Rsq'], times = times,
                            feat_name = feat_name)
            scipy.io.savemat(mat_name, mat_dict) 
#%%
# take the average of the regressed results  
n_times = 99 if not isMEG else 100
Rsq_all = np.zeros([n_feat, n_subj, n_vertices, n_times])
stc_path = stc_out_dir                                      
if False:        
    #================================================================================
    # save results in stc, this should be done by averageing across subjects 
    # 1. take the average of log10p across subjects 
    
    for j in range(n_feat):  
        feat_name = feat_name_seq[j]  
        for i in range(n_subj):
            subj = "Subj%d" %subj_list[i] 
            mat_name = mat_out_dir + "%s_%s_dim%d_dSPM_ols_ave_data_%s_morphed_%s.mat" \
                      %(subj, feat_name, n_dim, MEGorEEG[isMEG], fname_suffix)
            mat_dict = scipy.io.loadmat(mat_name)
            Rsq_all[j,i] = mat_dict['Rsq'][:,0:n_times]
        mean_Rsq_for_this_set = np.mean(Rsq_all[j], axis = 0)
        stc_Rsq = mne.SourceEstimate(data = mean_Rsq_for_this_set,vertices = vertices_to, 
                                    tmin = times[0], tstep = times[2]-times[1] )
        stc_name = stc_path + "%s_dim%d_dSPM_ols_ave_data_morphed_%s_%s" \
                  %(feat_name_seq[j], n_dim, MEGorEEG[isMEG], fname_suffix)
        stc_Rsq.save(stc_name) 
    
    '''
    # 2. take the difference
    #pair_list = [['AlexNet_fc6','AlexNet_conv1'],['neil_attr','AlexNet_conv1']]
    model_name = "CCA"
    feature_suffix = "pc6"
    pair_list = [['%s_fc7_%s' %(model_name, feature_suffix),
                     '%s_conv1_%s' %(model_name, feature_suffix)]]
    n_pair = len(pair_list)
    for k in range(n_pair):
        ind_list = list()
        for l in range(len(pair_list[k])):
            ind_list.append([i0 for i0 in range(n_feat) if feat_name_seq[i0] in [pair_list[k][l]]][0])
        diff = Rsq_all[ind_list[0]] - Rsq_all[ind_list[1]]
        mean_diff  = np.mean(diff, axis = 0)
        std_diff = np.std(diff, axis = 0)
        T_diff = mean_diff/std_diff*np.sqrt(n_subj)
        stc_T = mne.SourceEstimate(data = T_diff,vertices = vertices_to, 
                                    tmin = times[0], tstep = times[2]-times[1] )
        stc_name = stc_path + "diff_%s-%s_dim%d_mne_ols_ave_data_morphed_%s_%s" \
                  %( feat_name_seq[ind_list[0]], feat_name_seq[ind_list[1]], n_dim, MEGorEEG[isMEG], fname_suffix)
        stc_T.save(stc_name) 
    '''
        

if True:
    # render the results
    # This does not work in the spyder console, run it with the command line 
    suffix_list = feat_name_seq      
    fig_path =  data_root_dir0 +"Result_Mat/figs/source_reg/surface_view_%s/" %(MEGorEEG[isMEG])
    clim = dict(kind='value', lims=[2, 3, 6])
    #clim = dict(kind='value', lims = [0.01,0.02,0.05])
    subj = subject_to
    time_seq = range(0,90,2)
    time_seq = time_seq[0:30]
    surface = "inflated"
    for j in range(len(suffix_list)):
        stc_name = stc_path + "%s_dim%d_dSPM_ols_ave_data_morphed_%s_%s" \
                  %(feat_name_seq[j], n_dim, MEGorEEG[isMEG], fname_suffix) 
        stc = mne.read_source_estimate(stc_name)
        stc.data*= 100.0
        #'''
        brain = stc.plot(surface= surface, hemi='both', subjects_dir=subjects_dir,
                    subject = subject_to,  clim=clim, time_unit='ms',
                    size = 1200)
        for k in time_seq:
            brain.set_data_time_index(k)
            for view in ['ventral','dorsal','caud']:
                brain.show_view(view)
                im_name = fig_path + "dSPM_ave_%s_%s_Rsq_%03dms_%s_%s_%s.pdf" \
                   %(subj, suffix_list[j], np.int(np.round(stc.times[k]*1000)), 
                     view, MEGorEEG[isMEG], fname_suffix)
                brain.save_image(im_name)          
        brain.close()
        #'''
        
        for hemi in ['lh','rh']:
            brain = stc.plot(surface=surface, hemi= hemi, subjects_dir=subjects_dir,
                    subject = subj,  clim=clim, time_unit='ms')
            for k in time_seq:
                brain.set_data_time_index(k)
                for view in ['lateral','medial',#'ventral','dorsal'
                             ]:
                    brain.show_view(view)
                    im_name = fig_path + "dSPM_ave_%s_%s_Rsq_%03dms_%s_%s_%s_%s.pdf" \
                       %(subj, suffix_list[j], np.int(np.round(stc.times[k]*1000)), 
                         hemi, view, MEGorEEG[isMEG],fname_suffix)
                    brain.save_image(im_name)          
            brain.close()
            
 
           
    
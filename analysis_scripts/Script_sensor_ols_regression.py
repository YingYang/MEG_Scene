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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import time
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
#path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
path1 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
#path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from ols_regression import ols_regression
from Stat_Utility import bootstrap_mean_array_across_subjects


data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
data_root_dir =data_root_dir0 +"MEG_preprocessed_data/MEG_preprocessed_data/epoch_raw_data/"



#=======================================================================================
def sensor_space_regression_no_regularization(subj, ave_mat_path, X, 
                                              stats_model_flag = False):
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
        
    ave_data = ave_data[:,picks_all, :]
    # demean both X and neural data
    ave_data -= np.mean(ave_data, axis = 0)
    X -= np.mean(X, axis = 0)
    tmp_result = ols_regression(ave_data, X, stats_model_flag) 
    result = dict(Fval = tmp_result['F_val'], 
                  Rsq = tmp_result['Rsq'],
                  times = times, log10p =tmp_result['log10p'] )
    return result
#=======================================================================================
if __name__ == "__main__":
    
    #meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
    meg_dir = data_root_dir
    eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
    MEGorEEG = ['EEG','MEG']
    # try unsmoothed
    flag_swap_PPO10_POO10 = True
    MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect" \
        if flag_swap_PPO10_POO10 else "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"    
    
    model_name = "AlexNet"
    
    '''
    if True:
        isMEG = 0
        #Subj_list = ['SubjYY_100', 'SubjYY_200', 'SubjYY_500'] 
        subj_list = ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]
        n_times = 109
        n_channels = 128
        
    '''    
    Flag_CCA = False
    
    #for isMEG in [0, 1]:
    for isMEG in [1]:
        fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
        if isMEG:
            #isMEG = 1
            #Subj_list = range(1,10)
            Subj_list = range(1,19)
            n_times = 110
            n_channels = 306
        else:
            #isMEG = 0
            Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
            n_times = 109
            n_channels = 128
            
        n_Subj = len(Subj_list)
        n_im = 362
        
        
        regressor_dir = "/media/yy/LinuxData/yy_Scene_MEG_data/regressor/"
        mat_out_dir = "/media/yy/LinuxData/yy_Scene_MEG_data/Result_Mat/sensor_regression/ave_ols/"
    
        offset = 0.04 if isMEG else 0
        
        if not Flag_CCA:
            #feature_suffix = "no_aspect"
            #feature_suffix = "no_aspect_no_contrast100"
            
            # the one used in paper
            feature_suffix = "no_aspect_no_contrast160_all_im"
            #var_percent = 0.8
            n_dim = 10
            model_name = "AlexNet"
            #feat_name_seq = ["rawpixelgraybox"]
            feat_name_seq = []
            layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]
            #feat_name_seq = list()
            for l in layers:
                feat_name_seq.append("%s_%s_%s" %(model_name,l,feature_suffix))
            n_feat = len(feat_name_seq) 
            
            feat_name1 = layers
            
            X_list = []
            for j in range(n_feat):  
                # load the design matrix 
                feat_name = feat_name_seq[j]
                print feat_name
                regressor_fname = regressor_dir +  "%s_PCA.mat" %(feat_name)
                tmp = scipy.io.loadmat(regressor_fname)
                X0 = tmp['X'][0:n_im]
                suffix = "%d_dim" % n_dim
                print suffix, n_dim
                X = X0[:,0:n_dim]
                X_list.append(X)
                
            n_comp = n_dim
            """    
            for j in range(len(X_list)):
                X = X_list[j]
                feat_name = feat_name_seq[j]
                print feat_name
                for i in range(n_Subj):
                    t0 = time.time()
                    subj = "Subj%d" %Subj_list[i]
                    if isMEG:
                        ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
                    else:
                        ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
                    
                    result_reg = sensor_space_regression_no_regularization(subj, ave_mat_path, X)
                    mat_name = mat_file_out_dir + "%s_%s_result_reg_%s_%s.mat"  %(subj, MEGorEEG[isMEG],
                                                                                  feat_name, suffix)                
                    print n_dim
                    result_reg['n_dim'] = n_dim
                    scipy.io.savemat(mat_name, result_reg)  
             """
        else: 
            sep_CCA = True
            # CCA determined by the stimuli images
#            if sep_CCA == False:
#                n_comp = 15
#                fname = "%s_conv1_%s_%d_%s_fc6_%s_%d_intersect_%d.mat" %(model_name, feature_suffix,n_comp,\
#                               model_name, feature_suffix, n_comp, n_comp)
#                regressor_mat = scipy.io.loadmat(regressor_dir + fname)
#                sub_feat_seq = ['XresA','XresB','Xoverlap']
#                feat_name_seq = sub_feat_seq
#                n_feat = len(sub_feat_seq)
#                
#                X_list = list()
#                for j in range(n_feat):
#                    X_list.append(regressor_mat[sub_feat_seq[j]])
            # CCA determined by additional training data, only the common 10 dimensions of the first component is regressed out
            if sep_CCA == True:
                n_comp1 = 6
                n_comp = 6
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
            fname =  "/media/yy/LinuxData/yy_dropbox/Dropbox/" + \
                "/Scene_MEG_EEG/Features/" + \
                "Layer1_contrast/Stim_images_Layer1_contrast_noaspect.mat"
            mat_dict = scipy.io.loadmat(fname)
            tmp = mat_dict['contrast_noaspect']
            tmp = tmp- np.mean(tmp, axis = 0)
            U,D,V = np.linalg.svd(tmp)
            var_prop = np.cumsum(D**2)/np.sum(D**2)
            
            X_list.append(U[:,0:n_comp])
            feat_name_seq.append("contrast")
            n_feat = len(X_list)
            feat_name1 = ['res Layer 1','res Layer 7','common','local contrast']

        log10p = np.zeros([n_Subj, n_feat, n_channels, n_times])
        Rsq = np.zeros([n_Subj, n_feat, n_channels, n_times])
        for j in range(n_feat):
            X = X_list[j]
            for i in range(n_Subj):
                t0 = time.time()
                subj = "Subj%d" %Subj_list[i]
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
        if Flag_CCA:
            mat_name = mat_out_dir + "AlexNet_%s_CCA%d_ncomp%d_ave_%s.mat" %(MEGorEEG[isMEG], sep_CCA, n_comp, fname_suffix)
        else:
            mat_name = mat_out_dir + "AlexNet_%s_%s_ncomp%d_ave_%s.mat" %(MEGorEEG[isMEG], feature_suffix, n_comp, fname_suffix)
        
        print mat_name
        mat_dict = dict(Rsq = Rsq, times = times, log10p = log10p, Subj_list = Subj_list,
                        isMEG = isMEG, X_list = X_list, n_comp = n_comp, 
                        feat_name_seq = feat_name_seq, n_feat = n_feat,
                        feat_name1 = feat_name1)    
        scipy.io.savemat(mat_name, mat_dict)

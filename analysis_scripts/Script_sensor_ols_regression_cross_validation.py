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
'''
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
#from ols_regression import ols_regression
#from Stat_Utility import bootstrap_mean_array_across_subjects
'''

#=======================================================================================
def sensor_space_regression_no_regularization_train_test(ave_data, X, 
                                              train_ind,test_ind,
                                              times):
    """
    subj = "Subj2"
    ave_mat_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epochs_raw_data/"\
                 + "Subj2/%s_%s_ave.mat" %(subj, 1_50Hz_raw_ica_window_50ms)
    """
    
    assert( len(np.intersect1d(train_ind, test_ind)) == 0)
    assert ( len(np.unique(np.union1d(train_ind, test_ind))) == 362)
    
    n_im, m, n_times = ave_data.shape
    
    Rsq = np.zeros([m,n_times])
    Y_hat = np.zeros([len(test_ind), m, n_times])
    Y_truth = np.zeros([len(test_ind), m, n_times])
    
    X_aug = np.zeros([n_im, X.shape[1]+1])
    X_aug[:,0] = np.ones(n_im)
    X_aug[:,1::] = X
    X_aug_train = X_aug[train_ind,:]
    inv_op_train = (np.linalg.inv(X_aug_train.T.dot(X_aug_train))).dot(X_aug_train.T)
    
    
    for i in range(m):
        #if np.mod(i,1000) == 0:
        #    print "%f completed" % (np.float(i)/np.float(m)*100)
        for j in range(n_times):
            y_train = ave_data[train_ind,i,j]
            beta = np.dot(inv_op_train, y_train)
            
            # validate on the test set
            y_test = ave_data[test_ind,i,j]
            yhat_test = np.dot(X_aug[test_ind],beta)
            residual = y_test-yhat_test
            SSE = np.sum(residual**2)
            # total variance
            #TSS = np.sum((y_test - np.mean(y_test))**2)
            TSS = np.sum(y_test**2)
            Rsq[i,j] = 1.0-SSE/TSS
            # try variance instead of raw
            SSE1 = np.var(residual)
            TSS1 = np.var(y_test)
            # model explained variance
            Rsq[i,j] = 1.0-SSE1/TSS1
            Y_hat[:,i,j] = yhat_test
            Y_truth[:,i,j] = y_test

    result = dict(Rsq = Rsq, Y_hat = Y_hat, Y_truth = Y_truth, times = times)
    return result




#=======================================================================================
if __name__ == "__main__":
    
    data_root_dir = "/media/yy/LinuxData/yy_Scene_MEG_data/MEG_preprocessed_data/MEG_preprocessed_data/"
    #meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
    #eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
    #MEGorEEG = ['EEG','MEG']
    meg_dir = data_root_dir + "epoch_raw_data/"
    eeg_dir = "ToBeAdded"
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
    Flag_CCA = True
    #Flag_CCA = False
    
    isMEG = 1
    
    
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
    
    
    #%% load the regression design matrix
    
    #regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
    #mat_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/ave_ols/"
    #fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_reg/MEEG_no_aspect/"
    
    regressor_dir = "/media/yy/LinuxData/yy_Scene_MEG_data/regressor/"
    mat_out_dir = "/media/yy/LinuxData/yy_Scene_MEG_data/Result_Mat/sensor_regression/ave_ols/"
    
    offset = 0.04 if isMEG else 0
    
    if not Flag_CCA:
        #feature_suffix = "no_aspect"
        #feature_suffix = "no_aspect_no_contrast100"
        
        # this was the one actually used in the manuscript
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
        feat_name1 = layers

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
        
        feat_name1 = ['Res Layer 1', 'Res Layer 7','common','local contrast']



    n_features = len(X_list)

    
    
    
    n_folds = 18
   
    test_inds_seq = []
    tmp_ind0 = list(range(n_im))
    n_sample_per_fold = n_im//n_folds+1
    for k0 in range(n_folds):
        start = k0*n_sample_per_fold
        end = min((k0+1)*n_sample_per_fold, 362)
        if start < end:
            test_inds_seq.append(
                tmp_ind0[start:end])
            
        
        '''
        tmp_inds = np.arange(0,362).reshape([181,2])
        for i in range(tmp_inds.shape[0]):
            if np.random.randn(1)[0]>0:
                tmp_inds[i,:]= tmp_inds[i,::-1]
                
        train_test_inds = [tmp_inds[:,0], tmp_inds[:,1]]
        '''
        
    print (len(test_inds_seq))
    print (test_inds_seq)
        
    
    Rsq_all = np.zeros([n_folds, n_Subj, n_features, 
                        n_channels, n_times])
    relative_error = np.zeros([n_Subj, n_features, 
                        n_channels, n_times])
    
    from sklearn.model_selection import KFold
    
    
    
    # must do n-fold cross validation, no random ness!
    print ("number of folds = ")
    print (len(test_inds_seq))
    
   
    
    relative_error =  np.zeros([n_Subj, n_feat, n_channels, n_times])
    
    for i in range(n_Subj):
        
        t0 = time.time()
        subj = "Subj%d" %Subj_list[i]
        print ("subj %d" %i )
        
        # create the cv sequence for each subject
        kfold = KFold(n_splits = n_folds, shuffle = True)
        split = kfold.split(range(362))
        train_ind_list = []
        test_ind_list = []
        for (l, (train_index, test_index)) in enumerate (split):
            train_ind_list.append(train_index)
            test_ind_list.append(test_index)
            
        
        for j in range(n_feat):
            X = X_list[j]
            #subj = Subj_list[i]
            if isMEG:
                ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
            else:
                ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)

            mat_dict = scipy.io.loadmat(ave_mat_path)
            ave_data = mat_dict['ave_data']
            picks_all = mat_dict['picks_all'][0]
            times = mat_dict['times'][0]
    
            ave_data = ave_data[:,picks_all, :]
            # demean both X and neural data
            ave_data -= np.mean(ave_data, axis = 0)
            X -= np.mean(X, axis = 0)
    
    
            Y_truth_all = np.zeros([ n_im, n_channels, n_times])
            Y_hat_all = np.zeros([ n_im, n_channels, n_times])
            
            for l in range(n_folds):
                print ("CV %d feat %d subj %d" %(l,j,i))
                
        
                train_ind = train_ind_list[l]
                test_ind = test_ind_list[l]
                
                #print (train_ind)
                #print (test_ind)
            
  
                #test_ind = test_inds_seq[l]
                #train_ind = np.setdiff1d(range(362), test_ind)
                
                result = \
                    sensor_space_regression_no_regularization_train_test(
                            ave_data, X, train_ind,
                            test_ind, times)
                Rsq_all[l,i,j] = result['Rsq']
                Y_hat_all[test_ind] = result['Y_hat']
                Y_truth_all[test_ind] = result['Y_truth']
                print ("prediction written")
                times = result['times'][0:n_times]
                
                
                
            error = Y_hat_all-Y_truth_all
            relative_error[i,j] = np.sum(error**2, axis = 0)/np.sum(Y_truth_all**2, axis = 0)
            
    
    mean_relative_error = relative_error.mean(axis = 0) 
    Rsq_all_mean = Rsq_all.mean(axis = 0).mean(axis = 1)
    '''
    
    for toplot in [mean_relative_error, Rsq_all_mean]:
        plt.figure()
        for j in range(n_feat):
            plt.subplot(4,3,j+1)
            plt.imshow(toplot[j], vmin = None, vmax = None,
                       aspect = "auto");
            plt.colorbar();
            plt.title(feat_name_seq[j])
         
    '''
    
    # save the results as mat
    if Flag_CCA:
        mat_name = mat_out_dir + "AlexNet_%s_CCA%d_ncomp%d_ave_cv_%s.mat" %(MEGorEEG[isMEG], sep_CCA, n_comp, fname_suffix)
    else:
        mat_name = mat_out_dir + "AlexNet_%s_%s_ncomp%d_ave_cv_%s.mat" %(MEGorEEG[isMEG], feature_suffix, n_comp, fname_suffix)
    
    print (mat_name)
    mat_dict = dict(Rsq_all = Rsq_all.mean(axis = 0), 
                    relative_error = relative_error, 
                    times = times, Subj_list = Subj_list,
                    isMEG = isMEG, X_list = X_list, 
                    n_comp = n_comp, 
                    feat_name_seq = feat_name_seq, 
                    n_feat = n_feat, feat_name1 = feat_name1)    
    scipy.io.savemat(mat_name, mat_dict)

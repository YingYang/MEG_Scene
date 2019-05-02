# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import numpy as np
import matplotlib
import time
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

path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)

from ols_regression import ols_regression
import sklearn
import sklearn.linear_model
from sklearn import cross_validation



meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

#fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
MEG_fname_suffix = '1_110Hz_notch_ica_MEEG_match_ave_alpha15.0_no_aspect';
EEG_fname_suffix = '1_110Hz_notch_ica_PPO10POO10_swapped_MEEG_match_ave_alpha15.0_no_aspect';
MEGorEEG = ["EEG","MEG"] 

#Subj_list = range(1,19)
#n_Subj = len(Subj_list)
#Subj_has_EEG = np.ones(18, dtype = np.bool)
#Subj_has_EEG[[8,14,16]] = False


Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
n_Subj = len(Subj_list)
Subj_has_EEG  = np.ones([n_Subj], dtype = np.bool)

common_times = np.round(np.arange(-0.1, 0.8, 0.01), decimals = 2) 
n_times = len(common_times)
#MEG_offset = 0.04
MEG_offset = 0.02


nfold = 10
lr = sklearn.linear_model.LinearRegression()



fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_reg/"  

#n_comp_seq = [10,15,20,30,50,100,150,200,250,300]
n_comp_seq = [20,40,60,80] #,150,200,250,300]
for n_comp in n_comp_seq:
    Rsq = np.zeros(2, dtype = np.object)
    # cross validation error
    cv_error = np.zeros(2, dtype = np.object)
    n_im = 362
    for isMEG in [0,1]:
        fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
        n_channel = 306 if isMEG else 128
        tmp_data = np.zeros([n_Subj, n_im, n_channel, n_times] )
        tmp_Rsq = np.zeros([n_Subj, n_channel, n_times])
        tmp_cv_error = np.zeros([n_Subj, n_channel, n_times])
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
            tmp_data[j,:,:,:] = ave_data
        
        for j in range(n_Subj):  
            if isMEG:
                valid_subj = range(n_Subj)  
            elif Subj_has_EEG[j]:
                valid_subj = np.nonzero(Subj_has_EEG)[0]
            else:
                continue 
            tmp1 = tmp_data[j,:,:,:]
            other_subj = np.setdiff1d(valid_subj, j)
            tmp2 = tmp_data[other_subj,:,:,:]
            
            t0 = time.time()
            for t in range(n_times):
                tmp11 = tmp1[:,:,t]
                tmp22 = tmp2[:,:,:,t].transpose([1,0,2]).reshape([n_im, -1])
                u,d,v = np.linalg.svd(tmp22, full_matrices = False)
                regressor = u[:,0:n_comp]
                
                # obtain regression results
                tmp_result = ols_regression(tmp11[:,:, np.newaxis], regressor, stats_model_flag = False) 
                tmp_Rsq[j,:,t] = tmp_result['Rsq'][:,0]
                
                # cross_val_score use the score function of the classifier
                # score = coefficient of determination R^2 of the prediction
                # (1 - u/v), where u is the regression sum of squares ((y_true - y_pred) ** 2).sum()
                # and v is the residual sum of squares ((y_true - y_true.mean()) ** 2).sum().
                # should be able to match the rsq if not overfitted
                for l in range(tmp11.shape[1]):
                    tmp_cv_error[j,l,t] = np.mean(cross_validation.cross_val_score(lr, regressor, tmp11[:,l], cv = nfold))      
            
            print time.time()-t0
        Rsq[isMEG] = tmp_Rsq
        cv_error[isMEG] = tmp_cv_error
    
    
    mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/"\
             + "sensor_regression/ave_ols/Rsq_leave_one_subj_out_MEEG_%dpc.mat" \
             %n_comp
    print mat_name
    mat_dict = dict(Rsq = Rsq, cv_error = cv_error, common_times = common_times, n_times = n_times, 
                MEG_offset = MEG_offset, Subj_list = Subj_list, 
                Subj_has_EEG = Subj_has_EEG, n_comp = n_comp )
    scipy.io.savemat(mat_name, mat_dict)
    
    
    

if False:
    
    n_comp_seq = [20,40,60,80] #,150,200,250,300]
    N = len(n_comp_seq)
    n_method = 2
    n_times = 90
    cv_error_EEG = np.zeros([ N, n_Subj, 128, n_times])
    cv_error_MEG = np.zeros([ N, n_Subj, 306, n_times])
    for l in range(N):
        n_comp = n_comp_seq[l]
        mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/"\
           + "sensor_regression/ave_ols/Rsq_leave_one_subj_out_MEEG_%dpc.mat" \
           %n_comp
        mat_dict = scipy.io.loadmat(mat_name)
        cv_error_EEG[l] = mat_dict['cv_error'][0, 0]
        cv_error_MEG[l] = mat_dict['cv_error'][0, 1]
    
    common_offset = 0.02
    times = (mat_dict['common_times'][0]-common_offset)*1000.0
    cv_error_list = [cv_error_EEG, cv_error_MEG]
    MEGorEEG = ['EEG', 'MEG']
    
    plt.figure( figsize = (5, 3))
    for method_id in range(2):
        a = cv_error_list[method_id]
        # max across PC, mean across subject
        b = (a.max(axis = 0)).mean(axis = 0)
        print b.max()
        plt.plot(times, b.max(axis = 0)*100.0)
    
    plt.xlabel('times (ms)')
    plt.ylabel('maximum % variance explained')
    plt.legend(MEGorEEG)
    plt.tight_layout()
    plt.savefig(fig_outdir+ "sensor_reg_leave_on_subj_out.eps")
    
    
    

    
#==============================================================================

# do a box plot
if False:
    
    vmin, vmax = 0, 1.0
    plt.figure(figsize = (12,6))
    for isMEG in [0,1]:
        plt.subplot(1,2,isMEG+1);
        tmp = Rsq[isMEG].mean(axis = 0) if isMEG else Rsq[isMEG][Subj_has_EEG>0].mean(axis = 0)
        plt.imshow(tmp, aspect = "auto", interpolation = "none",
                   vmin = vmin, vmax = vmax, 
                   extent = [common_times[0], common_times[-1], 0, tmp.shape[0]],
                   origin = "lower");
        plt.colorbar(); plt.title(MEGorEEG[isMEG])
        plt.xlabel('time (s)'); plt.ylabel('channel index')
    plt.savefig(fig_outdir+ "sensor_reg_%dpc_loo_rsq.pdf" % n_comp)

    
    
    
    
    Rsq_max = np.zeros([2,len(n_comp_seq)])
    Rsq_max_no_baseline =  np.zeros([2,len(n_comp_seq)])
    cv_error = np.zeros([2, len(n_comp_seq)])
    for l in range(len(n_comp_seq)):
        tmp_mat = scipy.io.loadmat("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/"\
             + "sensor_regression/ave_ols/Rsq_leave_one_subj_out_MEEG_%dpc.mat" \
             %n_comp_seq[l])
        Rsq_max[0,l] = np.max(tmp_mat['Rsq'][0,0][Subj_has_EEG>0].mean(axis = 0))
        Rsq_max[1,l] = np.max(tmp_mat['Rsq'][0,1].mean(axis = 0))
    
        baseline_ind = tmp_mat['common_times'][0] < -0.01
        Rsq_max_no_baseline[0,l] = np.max(tmp_mat['Rsq'][0,0][Subj_has_EEG>0].mean(axis = 0)) - np.mean(tmp_mat['Rsq'][0,0][Subj_has_EEG>0].mean(axis = 0)[:, baseline_ind])
        Rsq_max_no_baseline[1,l] = np.max(tmp_mat['Rsq'][0,1].mean(axis = 0)) -np.mean(tmp_mat['Rsq'][0,1].mean(axis = 0)[:, baseline_ind])
        
        cv_error[0,l] =  np.max((tmp_mat['cv_error'][0,0][Subj_has_EEG>0]).mean(axis = 0))
        cv_error[1,l] =  np.max((tmp_mat['cv_error'][0,1]).mean(axis = 0))
    
    data_list =  [Rsq_max, Rsq_max_no_baseline, cv_error] 
    data_names = ['maxRsq','maxRsq_nobaseline','cv_error']
    
    for i in range(len(data_list)):
        plt.figure(figsize = (5,4))
        index = np.arange(1, len(n_comp_seq)+1)
        plt.plot(index, data_list[i].T*100.0, '+-')
        plt.xticks(index, n_comp_seq)
        plt.legend(['EEG', 'MEG'])
        plt.xlabel('number of principle components in regression')
        plt.ylabel('percentage of variance explained')
        plt.tight_layout()
        plt.savefig(fig_outdir+ "sensor_reg_loo_with_ncomp_%s.pdf" % data_names[i])
        
    
    
    
 

    
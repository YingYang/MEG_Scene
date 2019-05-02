# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import time
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io

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
import sklearn
import sklearn.linear_model
from sklearn import model_selection

data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
data_root_dir =data_root_dir0 +"MEG_preprocessed_data/MEG_preprocessed_data/epoch_raw_data/"

meg_dir = data_root_dir

#fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
MEG_fname_suffix = '1_110Hz_notch_ica_ave_alpha15.0_no_aspect';



Subj_list = range(1,19)
n_Subj = len(Subj_list)


common_times = np.round(np.arange(-0.1, 0.9, 0.01), decimals = 2) 
n_times = len(common_times)
#MEG_offset = 0.04
MEG_offset = 0.04


nfold = 10
lr = sklearn.linear_model.LinearRegression()

fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_reg/"  

#n_comp_seq = [10,15,20,30,50,100,150,200,250,300]
n_comp_seq = [ 6, 10,]
for n_comp in n_comp_seq:
    n_im = 362

    fname_suffix = MEG_fname_suffix 
    n_channel = 306 
    tmp_data = np.zeros([n_Subj, n_im, n_channel, n_times] )
    tmp_Rsq = np.zeros([n_Subj, n_channel, n_times])
    tmp_cv_error = np.zeros([n_Subj, n_channel, n_times])
    for j in range(n_Subj):
        subj = "Subj%d" %Subj_list[j]
        ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
        
        ave_mat = scipy.io.loadmat(ave_mat_path)
        offset = MEG_offset 
        times = np.round(ave_mat['times'][0] - offset, decimals = 2)
        time_ind = np.all(np.vstack([times <= common_times[-1], times >= common_times[0]]), axis = 0)
        ave_data = ave_mat['ave_data'][:, :, time_ind]
        ave_data -= ave_data.mean(axis = 0) 
        tmp_data[j,:,:,:] = ave_data
    
    # allow for some temporal shift
    #width = 2
    width = 2
    for j in range(n_Subj):  
        valid_subj = range(n_Subj)  
        
        tmp1 = tmp_data[j,:,:,:]
        other_subj = np.setdiff1d(valid_subj, j)
        tmp2 = tmp_data[other_subj,:,:,:]
        
        t0 = time.time()
        for t in range(n_times):
            tmp11 = tmp1[:,:,t]
            tmp_tmin, tmp_tmax = max(0, t-width), min(t+width+1,n_times)
            tmp22 = tmp2[:,:,:,tmp_tmin:tmp_tmax].transpose([1,0,2,3]).reshape([n_im, -1])
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
                tmp_cv_error[j,l,t] = np.mean(model_selection.cross_val_score(lr, regressor, tmp11[:,l], cv = nfold))      
        
        print (time.time()-t0)
        print (j, n_comp)
    Rsq = tmp_Rsq
    cv_error = tmp_cv_error
    
    
    mat_name = data_root_dir0 + "/Result_Mat/"\
             + "sensor_regression/ave_ols/Rsq_leave_one_subj_out_MEEG_%dpc.mat" \
             %n_comp
    print (mat_name)
    mat_dict = dict(Rsq = Rsq, cv_error = cv_error, common_times = common_times, n_times = n_times, 
                MEG_offset = MEG_offset, Subj_list = Subj_list, 
                n_comp = n_comp )
    scipy.io.savemat(mat_name, mat_dict)
    
    
    
#%%
if False:
    
    n_comp_seq = [6, 10, 20,40 ,80] #,150,200,250,300]
    N = len(n_comp_seq)
    n_method = 2
    n_times = 100
    cv_error_MEG = np.zeros([ N, n_Subj, 306, n_times])
    for l in range(N):
        n_comp = n_comp_seq[l]
        mat_name = data_root_dir0 + "Result_Mat/"\
           + "sensor_regression/ave_ols/Rsq_leave_one_subj_out_MEEG_%dpc.mat" \
           %n_comp
        mat_dict = scipy.io.loadmat(mat_name)
        cv_error_MEG[l] = mat_dict['cv_error']
    
    common_offset = 0.0
    times = (mat_dict['common_times'][0]-common_offset)*1000.0
    cv_error_list = [cv_error_MEG]
    
    plt.figure( figsize = (5, 3))
    a = cv_error_MEG
    b = a.mean(axis = 1).max(axis = 1)
    # max across PC, mean across subject
    plt.plot(b.T*100);
    legends = []
    for l in n_comp_seq:
        legends.append("p0 = %d" %l)
    plt.legend(legends)
    plt.ylim(-40, 40)
    
    plt.grid()
    plt.xlabel('times (ms)')
    plt.ylabel('maximum % variance explained')
    plt.tight_layout()
    plt.savefig( data_root_dir0 + "/Result_Mat/figs/sensor_reg/"+ "max_sensor_reg_leave_on_subj_out.eps")
    plt.savefig( data_root_dir0 + "/Result_Mat/figs/sensor_reg/"+ "max_sensor_reg_leave_on_subj_out.png")
    
    
    from Script_sensor_results_visualization import visualize_sensors_MEG_topo
    import mne
    fname = "/media/yy/LinuxData/yy_Scene_MEG_data/MEG_preprocessed_data/MEG_preprocessed_data/epoch_raw_data/Subj1/Subj1_run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"
    tmp_epoch = mne.read_epochs(fname)
    info1 = tmp_epoch.info
    del(tmp_epoch)
    
    tmp = cv_error_MEG[0,:,:].mean(axis = 0)
    vmin = -0.15
    vmax = 0.4
    
    plt.figure()
    plt.plot(times, tmp.max(axis = 0),);
    plt.ylim(  [vmin, vmax])
    figname =  fig_outdir +  "MEG_sensor_reg_leave_on_subj_out_p%d_max" %(n_comp_seq[0])     
    plt.xlabel('time (ms)')
    plt.ylabel('max % variance across sensors')
    plt.savefig(figname+".png")
    plt.savefig(figname+".eps")
    
    
    time_array = np.arange(0.05,0.75,0.05)
    fig_outdir = data_root_dir0 + "Result_Mat/figs/sensor_reg/MEEG_no_aspect/"
    figsize = (20,10)
    figname =  fig_outdir +  "MEG_sensor_reg_leave_on_subj_out_p%d" %(n_comp_seq[0])       
    visualize_sensors_MEG_topo( info1, tmp*100, vmin*100, vmax*100, figname, figsize, 
                      times/1000.0, time_array, title = "noise ceiling", ylim = [vmin*100, vmax*100])

    
#==============================================================================


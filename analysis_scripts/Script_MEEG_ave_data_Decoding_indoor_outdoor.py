# -*- coding: utf-8 -*-
"""
20160616
Previouly, to compute power, I used H1 = results of the data, and two rounds of permutation, 
one for test, one for computing the proportion of rejects. This is wrong!
Probably there is not a good way of computing power from empirical data. 

So this script is modified as to either compute sharp ratio ( testing value/ quantile of permutation) 
or the T-statistics via bootstrap!
Actually bootstrap would not work for RSM, because of the duplicates
"""

import mne, time
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io
import time

import sklearn
from sklearn import cross_validation




import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)

import pandas as pd



#============= load the category labels from the csv, use the newest one
# load the csv
category_csv = '/home/ying/Dropbox/Scene_MEG_EEG/Features/StimSUN_semantic_feat/'+ \
            'MEG_NEIL_Image_SUN_hierarchy_manual_20160617.csv'
sun_hierarchy = pd.read_csv(category_csv, delimiter = ',',
                            skiprows = 0, usecols = np.arange(1,20))

# indoor, out door natrual, out door manmade
labels = sun_hierarchy.values[:,0:3]
print labels.sum(axis = 0)

# remove the outdoor natural class, 
to_include = np.all( np.vstack( [ labels.sum(axis = 1) == 1,
                                 labels[:,1] == 0]), axis = 0)

label_in_out = labels[to_include == True][:,[0,2]]


#%% data paths
meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
fname_suffix = '1_110Hz_notch_ica_MEEG_match_ave_alpha15.0_no_aspect';
subj_list = np.hstack([ np.arange(1,9), np.arange(10,15) ]);
n_subj = len(subj_list);

#%% parameters ===========
offset = 0.02
alpha = 0.05

tmin, tmax = 0.0, 0.5

n_times = 109 
metric = "correlation"

#%% 
im_proportion = np.array([0.1,0.2,0.3,0.4,0.5,0.7,0.9])
n_im = 362
n_im_seq = np.zeros(len(im_proportion), dtype = np.int)
n_proportion = len(n_im_seq)
for l in range(n_proportion):
    n_im_seq[l] = np.int(np.floor(im_proportion[l]*n_im))
# if 0, do permutation, if 1, do bootstrap
perm_or_btstrp = 0
perm_or_btstrp_string = ['perm','btstrp']
#perm_or_btstrp = 1

# number of permutations or number of bootstraps
B = 100
# number of subsampling given a fixed number of samples
n_rep_per_n_im = 30
# generate sub-sample sequence or bootstrap sequence
select_im_id = np.zeros([n_proportion, n_rep_per_n_im], dtype = np.object)
for l in range(n_proportion):
    tmp_orig = np.arange(n_im_seq[l])
    for l1 in range(n_rep_per_n_im):
        select_im_id[l,l1] = (np.random.choice(n_im, n_im_seq[l], replace = False)).astype(np.int)

# which is with respect to the selected images
perm_or_btstrp_seq = np.zeros([n_proportion, n_rep_per_n_im, B], dtype = np.object)
for l in range(n_proportion):
    tmp_orig = np.arange(n_im_seq[l])
    for l1 in range(n_rep_per_n_im):
        for l2 in range(B):
            if perm_or_btstrp ==0:
                perm_or_btstrp_seq[l,l1,l2] = (np.random.permutation(tmp_orig)).astype(np.int)
            else:
                perm_or_btstrp_seq[l,l1,l2] = (np.random.choice(n_im_seq[l], n_im_seq[l], replace = True)).astype(np.int)               
                   
#===================================================
# just save everything, and compute things afterwards       
# n_subj, n_modality, n_proportion, n_rep_per_n_im, B+1, n_times
corr_ts_all = np.zeros([n_subj, 2, n_proportion, n_rep_per_n_im, B+1, n_times]) 
          
for i in range(n_subj):
    # one Subject
    subj = "Subj%d" %subj_list[i]
    mat_path = [eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix),
                meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)]
    modality = ['EEG','MEG']
    for l in range(len(mat_path)):
        data_dict = scipy.io.loadmat(mat_path[l])
        ave_data = data_dict['ave_data']
        ave_data = ave_data[to_include]
        ave_data -= np.mean(ave_data)
        
        time_ms = (data_dict['times'][0]*1000).astype(np.int)
        # should I equilize the units of magnetometers and gradiometers?
        if modality[l] is 'MEG':
            ave_data = np.reshape(ave_data, [ave_data.shape[0],102,3,len(time_ms)])
            # compute the std for each channel type
            ave_data_transpose = (np.transpose(ave_data, [2,1,0,3])).reshape([3,-1])
            std = np.std(ave_data_transpose, axis = 1)
            for ll in range(3):
                ave_data[:,:,ll,:] /= std[ll]
            ave_data = np.reshape(ave_data,[ave_data.shape[0],306,len(time_ms)] )
        
            tmp_times =  data_dict['times'][0]-offset
            
        #time_ind = np.all( np.vstack([ tmp_times >= tmin, tmp_times <= tmax]) , axis = 0)
        #ave_data = ave_data[:,:, time_ind]
        tmp_label = label_in_out[:,0]
        tmp_n_im = len(tmp_label) 
        
        # do an anova conncatenated classifier
        lr = sklearn.linear_model.LogisticRegression(C = 0.01)
        cv_score_ts = np.zeros(n_times)
        for t in range(n_times):
            tmp_X = scipy.stats.zscore(ave_data[:,:,t]) 
            tmp_cv_score = cross_validation.cross_val_score(lr, tmp_X, tmp_label, cv = 100)
            cv_score_ts[t] = tmp_cv_score.mean()
            
        
        
        # to be modified        
        for k1 in range(n_proportion):
            for k2 in range(n_rep_per_n_im):
                tmp_im_set = select_im_id[k1,k2]
                tmpX = X[tmp_im_set].copy()
                tmp_data = ave_data[tmp_im_set].copy()
                if perm_or_btstrp == 0:
                    # permutation
                    tmp_perm_seq_obj = perm_or_btstrp_seq[k1,k2]
                    tmp_perm_seq = np.zeros([B, len(tmp_perm_seq_obj[0])])
                    for b in range(B):
                        tmp_perm_seq[b] = tmp_perm_seq_obj[b]
                    result_rdm = get_rsm_correlation(tmp_data, tmpX, n_perm = B,
                                 perm_seq = tmp_perm_seq, metric = metric,
                                 demean = True, alpha = 0.05) 
                    corr_ts_all[i,l,k1,k2,0,:] =  result_rdm['corr_ts'][0:n_times]
                    corr_ts_all[i,l,k1,k2,1::,:] =  result_rdm['corr_ts_perm'][:,0:n_times]
                else:
                    # Actually, due to the duplication, bootstrap results are weird, 
                    # they are not centered around corrts
                    # bootstrap
                    result_rdm0 = get_rsm_correlation(tmp_data, tmpX, n_perm = 0,
                                 perm_seq = None, metric = metric,
                                 demean = True, alpha = 0.05)
                    corr_ts_all[i,l,k1,k2,0,:] =  result_rdm['corr_ts'][0:n_times]
                    # something is wrong here, to be corrected
                    for b in range(B):
                        tmp_btstrp_seq = perm_or_btstrp_seq[k1,k2,b]
                        tmpX_bt = tmpX[tmp_btstrp_seq].copy()
                        tmp_data_bt = tmp_data[tmp_btstrp_seq].copy()
                        result_rdm_bt = get_rsm_correlation(tmp_data_bt, tmpX_bt, n_perm = 0,
                                 perm_seq = None, metric = metric,
                                 demean = True, alpha = 0.05)
                        corr_ts_all[i,l,k1,k2,b+1,:] = result_rdm_bt['corr_ts'][0:n_times]
                        

# save the results in mat file
mat_name =  "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/"\
    + "sensor_dependence/comp_MEG_EEG/%s_rsm_corrts_%s.mat" %(feat_name, perm_or_btstrp_string[perm_or_btstrp])
mat_dict = dict( im_proportion = im_proportion, n_proportion = n_proportion, 
                n_rep_per_n_im = n_rep_per_n_im,  B = B,
                perm_or_btstrp = perm_or_btstrp,  perm_or_btstrp_seq = perm_or_btstrp_seq,
                select_im_id = select_im_id,  subj_list = subj_list, 
                corr_ts_all = corr_ts_all)   
scipy.io.savemat(mat_name, mat_dict)        
                   
 
   


# the common time here has a problem, some times are missing

if False:
    import scipy.io
    import numpy as np
    n_subj = len(subj_list)
    n_times = 87
    # not useful, it is not power
    if False:
        power_mat = np.zeros([n_subj, 2, len(im_proportion), n_times])
        for i in range(n_subj):
            subj = "Subj%d" %subj_list[i]
            mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/"\
                 + "sensor_dependence/comp_MEG_EEG/%s_%s_rsm_corr_power.mat" %(subj, X_id)
            mat_dict = scipy.io.loadmat(mat_name)
            power = mat_dict['power']
            power_ave = np.mean(power, axis = 2)
            common_time = mat_dict['common_time'][0]
            im_proportion = mat_dict['im_proportion'][0]
            power_mat[i] = power_ave
            del(mat_dict)
    mean_power_across_subj = np.mean(power_mat, axis = 0)
    #==========(corr_ts- np.max(quantile(0.925) perm))/np.max(quantile(0.925) perm))
    normalized_ts = np.zeros([n_subj, 2, len(im_proportion), n_times])
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/"\
             + "sensor_dependence/comp_MEG_EEG/%s_%s_rsm_corr_power.mat" %(subj, X_id)
        mat_dict = scipy.io.loadmat(mat_name)
        corr_ts = mat_dict['corr_ts']
        #[2, len(im_proportion), n_ave, n_perm2, n_perm, n_times]
        corr_ts_perm = np.reshape(mat_dict['corr_ts_perm'],[2, len(im_proportion), n_ave, n_perm2*n_perm, n_times])
        # get the percentile
        percentile = np.percentile(corr_ts_perm, (1-alpha/2.0)*100, axis = 3)
        max_percentile = np.max(percentile, axis = -1)
        tmp = ((corr_ts.transpose([3,0,1,2]) - max_percentile)/max_percentile).mean(axis = -1)
        tmp = tmp.transpose([1,2,0])
        normalized_ts[i] = tmp
    
    # offset in time
    common_time = mat_dict['common_time'][0]-20
    #=================================================================
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/meg_eeg_sensor/"

    #data = mean_power_across_subj
    #vmin, vmax = 0,1.0
    #data_name = "empi_power"
    
    data = np.mean(normalized_ts, axis = 0)
    vmin, vmax = 0,8.0
    data_name = "normalized_corr"
    
    
    plt.figure(figsize = (12,6))
    xticks = np.arange(0,n_times,10)
    MEGorEEG = ['MEG','EEG']
    for j in range(2):
        _=plt.subplot(1,2,j+1)
        _=plt.imshow(data[j], interpolation = "none", aspect = "auto", 
                   vmin = vmin, vmax = vmax)
        _=plt.title(MEGorEEG[j])
        _=plt.xlabel('time (ms)')
        _=plt.ylabel('proportion %')
        _=plt.colorbar()
        _=plt.xticks(xticks, common_time[xticks])
        _=plt.yticks(range(len(im_proportion)), im_proportion)
    plt.tight_layout()
    plt.savefig(fig_outdir + "Subj_all_MEEG_RSM_corr_feat_%s_%s.pdf" %(X_id, data_name))
    
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    NCURVES = 10 
    values = range(NCURVES)
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    # plot conv1, conv2, conv5 and fc6 in the same figure
    colorVal_list = list()
    for j in (im_proportion*10-1).astype(np.int):
        colorVal = scalarMap.to_rgba(values[j])
        colorVal_list.append(colorVal)
    
    plt.figure(figsize = (12,6))
    xticks = np.arange(0,n_times,10)
    MEGorEEG = ['MEG','EEG']
    for j in range(2):
        _=plt.subplot(1,2,j+1)
        for l in range(len(im_proportion)):
            _=plt.plot(common_time, data[j,l], color = colorVal_list[l])
        _=plt.title(MEGorEEG[j])
        _=plt.xlabel('time (ms)')
        _=plt.ylim(-2,7)
        _=plt.grid()
        _=plt.legend(im_proportion)
    plt.savefig(fig_outdir + "Subj_all_MEEG_RSM_corr_feat_%s_%s_plots.pdf" %(X_id, data_name))
    
    
    
    
    plt.figure( figsize = (15,30))
    for i in range(n_subj):
        for j in range(2):
           _=plt.subplot(n_subj, 2, 2*i+j+1)
           _=plt.imshow(normalized_ts[i,j], interpolation = "none", aspect = "auto", 
               vmin = vmin, vmax = vmax)
           _=plt.title(MEGorEEG[j] + " Subj%d" %subj_list[i])
           _=plt.xlabel('time (ms)')
           _=plt.ylabel('proportion %')
           _=plt.colorbar()
           _=plt.xticks(xticks, common_time[xticks])
           _=plt.yticks(range(len(im_proportion)), im_proportion)
    #plt.tight_layout() 
    plt.savefig(fig_outdir + "Subj_indiv_MEEG_RSM_corr_feat_%s_%s.pdf" %(X_id,data_name))
    

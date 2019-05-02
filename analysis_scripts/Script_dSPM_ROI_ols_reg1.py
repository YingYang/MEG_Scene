import numpy as np
import scipy.io

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
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
from ols_regression import ols_regression
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from Stat_Utility import bootstrap_mean_array_across_subjects

#ROI_bihemi_names = [ 'pericalcarine', 
#                    'PPA_c_g', 'TOS_c_g', 'RSC_c_g', 'LO_c_g',
#                    'inferiortemporal', 'lateraloccipital', 'fusiform',
#                    'insula', 'lateralorbitofrontal',  'medialorbitofrontal']

ROI_bihemi_names = [ 'pericalcarine',  'LO_c_g']
nROI = len(ROI_bihemi_names)                    
labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"  
MEGorEEG = ['EEG','MEG']
isMEG = True
# For now for MEG only
stc_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/" \
            + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]
fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    
times = np.arange(0.01,0.96,0.01)
n_times = 100
times_in_ms = times*1000.0

# load the design matrix
#subj_list = [1,2,3,4,5,6,7,8,9,10,12,13] 
subj_list = [1,2,3,4,5,6,7,8,9,10,12,13]   
#subj_list = np.arange(1,14)
n_subj = len(subj_list)
regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"

  
if False:
    n_im = 362
    var_percent = 200
    model_name = "AlexNet"
    feat_name_seq = []
    layers = ["conv1","fc6"]
    feature_suffix = "no_aspect"
    for layer in layers:
        feat_name_seq.append("%s_%s_%s" %(model_name, layer, feature_suffix))
    n_feat_name = len(feat_name_seq)
    X= np.zeros([n_im, var_percent*2])
    for j in range(n_feat_name):  
        # load the design matrix 
        feat_name = feat_name_seq[j]
        print feat_name
        regressor_fname = regressor_dir +  "%s_PCA.mat" %(feat_name)
        tmp = scipy.io.loadmat(regressor_fname)
        X0 = tmp['X']
        if var_percent > 1:
            n_dim = np.int(min(X0.shape[1], var_percent))
            suffix = "%d_dim" % n_dim
        else:
            n_dim = np.nonzero(tmp['var_per_cumsum'][0] >=var_percent)[0][0]
            suffix = "%d_percent" % np.int(var_percent*100)
        
        print suffix, n_dim
        X[:,j*n_dim:(j+1)*n_dim] = X0[:,0:n_dim]

if True:    
    
    mode_id = 2
    mode_names = ['','Layer1_6','localcontrast_Layer6']
    #feat_name = ['conv1','fc6']
    feat_name = ['localcontrast','fc6']

    n_comp = 4
    
    X_list = list()
    for j in range(2):
        #tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_cca_%d_residual_svd.mat" %(feat_name[j], n_comp))
        tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_%s_cca_%d_residual_svd.mat" \
                        %(mode_names[mode_id], feat_name[j], n_comp))
        X_list.append(tmp['u'][:,0:n_comp])
    
    X_list.append(tmp['merged_u'][:,0:n_comp])
    feat_name.append( "CCA_merged")
    n_sub_feat = len(X_list)
    pairs = [[1,0]]  
    n_pair = len(pairs) 
    pair_names = ['Layer6-Layer1']
    X_name = feat_name
        
    
    #X = np.hstack([X_list[0],X_list[1]])
    #X = np.hstack([X,X_list[2]])
    
    tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_%s_cca_%d_residual_svd.mat" \
                        %(mode_names[mode_id], feat_name[0], n_comp))
    ind = [0,10,20]
    X = tmp['u'][:,ind]
    Xnames = tmp['d'][:,ind]
    


# add L2 penalization
Lambda = 0.0
inv_op = (np.linalg.inv(X.T.dot(X) + np.eye(X.shape[1])*Lambda)).dot(X.T)
    
n_feat = 3
beta_hi_lo = np.zeros([n_subj, nROI, n_times])
beta_stat_all = np.zeros([n_subj, nROI, n_feat, n_times])
for i in range(n_subj):        
    subj = "Subj%d" %subj_list[i]
    labeldir1 = labeldir + "%s/" % subj
    # load and merge the labels
    labels_bihemi = list()
    for j in ROI_bihemi_names:
        tmp_label_list = list()
        for hemi in ['lh','rh']:
            print subj, j, hemi
            tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,j,hemi)
            tmp_label = mne.read_label(tmp_label_path)
            tmp_label_list.append(tmp_label)
        labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1]) 
    fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                           + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
    fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
    src = fwd['src']
    ROI_ind = list()
    for j in range(nROI):
        tmp_label = labels_bihemi[j]
        _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
        ROI_ind.append(tmp_src_sel)
        # load the source solution
        mat_dir = stc_out_dir + "%s_%s_%s_ave.mat" %(subj, MEGorEEG[isMEG],fname_suffix)
        mat_dict = scipy.io.loadmat(mat_dir)
        source_data = mat_dict['source_data']
        times = mat_dict['times'][0]
        n_times = len(times)
        del(mat_dict)
        
    ROI_data = list()
    for j in range(nROI):
        tmp = source_data[:,ROI_ind[j],:]
        tmp -= np.mean(tmp, axis = 0)
        ROI_data.append(tmp)
    del(source_data)

    for j in range(nROI): 
        tmp_data = ROI_data[j]
        beta = np.zeros([tmp_data.shape[1], tmp_data.shape[2], X.shape[1]] )
        for i0 in range(tmp_data.shape[1]):
            for j0 in range(tmp_data.shape[2]):
                y = tmp_data[:,i0,j0]
                tmp_beta = np.dot(inv_op, y)
                beta[i0,j0,:] =tmp_beta
        beta_reshape = np.reshape(beta, [tmp_data.shape[1], tmp_data.shape[2], n_feat,X.shape[1]//n_feat])
        beta_stat = np.sqrt(np.sum(beta_reshape**2, axis = 3))
        beta_stat = np.mean(beta_stat, axis = 0)
        beta_stat_all[i,j] = beta_stat[0:n_times,:].T
        beta_hi_lo[i,j] = (beta_stat[:,1]**2 -beta_stat[:,0]**2)

#================================================================================
plt.figure(figsize = (10,10))
#n1,n2 = 4,3
n1, n2 = 2, nROI//2+1

col = ['b','g','r']
for j in range(nROI):
    ax = plt.subplot(n1,n2,j+1)   
    for l in range(n_feat):
        tmp = bootstrap_mean_array_across_subjects(beta_stat_all[:,j,l], alpha = 0.05/n_times/nROI)
        tmp_mean = tmp['mean']
        tmp_se = tmp['se']
        ub = tmp['ub']
        lb = tmp['lb'] 
        _ = ax.plot(times, tmp_mean, color = col[l])
        _ = ax.fill_between(times, ub, lb, alpha=0.4, facecolor = col[l]) 
        _ = plt.title(ROI_bihemi_names[j])


plt.legend(Xnames, loc = 9)

#    
#for j in range(nROI):
#    ax = plt.subplot(n1,n2,j+1)  
#    tmp = bootstrap_mean_array_across_subjects(beta_hi_lo[:,j,:], alpha = 0.05/n_times/nROI)
#    tmp_mean = tmp['mean']
#    tmp_se = tmp['se']
#    ub = tmp['ub']
#    lb = tmp['lb'] 
#    _ = ax.plot(times, tmp_mean)
#    _ = ax.fill_between(times, lb, ub,  alpha=0.4) 
#    #_ = plt.title(ROI_bihemi_names[j] + "\n" + pair_names[l])
#    _ = ax.plot(times, np.zeros(times.shape), 'k')
#    _ = plt.title(ROI_bihemi_names[j])
   

    

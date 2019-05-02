import mne
import numpy as np
import scipy.io
import copy

mne.set_log_level('WARNING')
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
from ols_regression import ols_regression
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from Stat_Utility import bootstrap_mean_array_across_subjects






meta_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"
MEG_dir = meta_dir + "MEG_DATA/DATA/epoch_raw_data/"
EEG_dir = meta_dir + "EEG_DATA/DATA/epoch_raw_data/"
save_name_suffix = "1_110Hz_notch_ica"


mat_name = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/aspect_energy.mat"
mat_dict = scipy.io.loadmat(mat_name)
aspect_X = mat_dict['X']

isMEG = False
MEGorEEG = ['EEG','MEG']
if isMEG:
    subj_list = np.arange(1,19)
else:
    subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]    

n_subj = 12
#n_subj = len(subj_list)



fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
fname_suffix1 = "1_110Hz_notch_ica_no_aspect"

epoch_dir = MEG_dir if isMEG else EEG_dir
stc_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/" \
                + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]

              
for i in range(n_subj):
    subj = "Subj%d" %(subj_list[i]) 
    if isMEG:
        mat_name = epoch_dir + "%s/%s_%s_repeated_trials.mat" %(subj, subj, save_name_suffix)
    else:
        mat_name = epoch_dir + "%s_EEG/%s_%s_repeated_trials.mat" %(subj, subj, save_name_suffix)
    mat_dict = scipy.io.loadmat(mat_name)
    epoch_mat_repeat = mat_dict['epoch_mat_repeat']
    RT_repeat = mat_dict['RT_repeat'][0]
    im_id_repeat = mat_dict['im_id_repeat'][0]
    times =  mat_dict['times'][0]
    
    n_trials = epoch_mat_repeat.shape[0]
    #========= regress out the aspect ratios
    tmp_aspect_X = aspect_X[im_id_repeat.astype(np.int)]
    proj = np.eye(n_trials) - reduce(np.dot, [tmp_aspect_X, 
            np.linalg.inv(tmp_aspect_X.T.dot(tmp_aspect_X)), tmp_aspect_X.T])
    for t in range(len(times)):
        epoch_mat_repeat[:,:,t] = proj.dot(epoch_mat_repeat[:,:,t])
    
    #========= source localization 
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
        epochs_path = ("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                     +"EEG_DATA/DATA/epoch_raw_data/%s_EEG/%s_EEG_%s" \
                    %(subj, subj, "filter_1_110Hz_notch_ica_reref_smoothed-epo.fif.gz"))
        datapath = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                  + "EEG_DATA/DATA/epoch_raw_data/%s_EEG/%s_EEG_%s.mat" \
                  %(subj, subj, fname_suffix)
                      
               
    # load data
    epochs = mne.read_epochs(epochs_path)
    if isMEG:
        epochs1 = mne.epochs.concatenate_epochs([epochs, copy.deepcopy(epochs)])
        epochs = epochs1[0:n_trials]
        epochs._data = epoch_mat_repeat.copy()
        del(epochs1)
    else:
        epochs = epochs[0:n_trials]
        epochs._data = epoch_mat_repeat.copy()

    # a temporary comvariance matrix
    # note, here tmax was different from the averaged data, there were too few examples,
    # so we expended it to -0.01
    cov = mne.compute_covariance(epochs, tmin=None, tmax=-0.01)
    # load the forward solution
    fwd = mne.read_forward_solution(fwd_path, surf_ori = True)
    # create inverse solution
    inv_op = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, depth = 0.8,
                                                    fixed = True)
    stc = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2 = 1.0, method = "dSPM")
    
    source_data = np.zeros([n_trials, stc[0].data.shape[0], stc[0].data.shape[1]])
    for j in range(n_trials):
        source_data[j] = stc[j].data 
        
    times = epochs.times
    times_ind = times>=0.0
    times = times[times_ind]
    source_data = source_data[:,:, times_ind]
    del(stc)

    mat_name = stc_out_dir + "%s_%s_%s_repeated.mat" %(subj, MEGorEEG[isMEG],fname_suffix1)
    # no offset is applied here
    time_corrected = 0
    mat_dict = dict(source_data = source_data, times = times, time_corrected = time_corrected)
    scipy.io.savemat(mat_name, mat_dict)


#========================================================        


ROI_bihemi_names = ['pericalcarine']
                    #'lateralorbitofrontal',  'medialorbitofrontal']                   
labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"  

regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"  
n_comp = 10
feat_name = ['conv1','fc6']
X_list = list()
for j in range(2):
    tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_cca_%d_residual_svd.mat" %(feat_name[j], n_comp))
    X_list.append(tmp['u'][:,0:n_comp])

X_list.append(tmp['merged_u'][:,0:n_comp])
feat_name.append( "CCA_merged")
n_sub_feat = len(X_list)
pairs = [[1,0],[0,2],[1,2]]  
n_pair = len(pairs) 
pair_names = ['Layer6-Layer1','Layer1-Common','Layer6-Common']
X_name = feat_name

ROI_partition_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/" 
n_part = 3
quantile_flag = False
nROI0 = len(ROI_bihemi_names)
for l in range(n_part):
    ROI_bihemi_names.append("vent_%d_qtl%d" %(l, quantile_flag))  
nROI = len(ROI_bihemi_names)    


#================== actual regression =========================  
n_times = 99
n_reg = len(X_list)
val = np.zeros([n_subj,nROI,n_reg, n_times])
val_normalized = np.zeros([n_subj,nROI,n_reg, n_times])
val_normalized_timewise = np.zeros([n_subj,nROI,n_reg, n_times])


for i in range(n_subj):
    subj = "Subj%d" %subj_list[i]
    labeldir1 = labeldir + "%s/" % subj
    # load and merge the labels
    labels_bihemi = list()
    for j in range(nROI0):
        tmp_name = ROI_bihemi_names[j]
        tmp_label_list = list()
        for hemi in ['lh','rh']:
            print subj, tmp_name, hemi
            tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,tmp_name,hemi)
            tmp_label = mne.read_label(tmp_label_path)
            tmp_label_list.append(tmp_label)
        labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1]) 
    if isMEG:
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                           + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
    else:
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                           + "EEG_DATA/DATA/fwd/%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)
    fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
    src = fwd['src']
    ROI_ind = list()
    for j in range(nROI0):
        tmp_label = labels_bihemi[j]
        _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
        ROI_ind.append(tmp_src_sel)
   
    # add ROI indices
    stc_name = ROI_partition_dir + "%s/%s_ventral_divide%d_label_quantile%d" %(subj, subj, n_part, quantile_flag)
    tmp_stc = mne.source_estimate.read_source_estimate(stc_name)
    dipole_label_val = tmp_stc.data[:,0]
    for i0 in range(n_part):
        tmp_ind = np.all( np.vstack([ dipole_label_val >= i0+0.5, dipole_label_val < i0+1.5]), axis = 0)
        ROI_ind.append(np.nonzero(tmp_ind)[0])
    del(stc_name, tmp_stc, dipole_label_val)    
        
    # load the source solution
    mat_dir = stc_out_dir + "%s_%s_%s_repeated.mat" %(subj, MEGorEEG[isMEG],fname_suffix1)
    mat_dict = scipy.io.loadmat(mat_dir)
    source_data = mat_dict['source_data']
    times = mat_dict['times'][0][0:n_times]
    del(mat_dict)
        
    ROI_data = list()
    for j in range(nROI):
        tmp = source_data[:,ROI_ind[j],:]
        tmp -= np.mean(tmp, axis = 0)
        ROI_data.append(tmp)
    del(source_data)
    
    if isMEG:
        mat_name1 = epoch_dir + "%s/%s_%s_repeated_trials.mat" %(subj, subj, save_name_suffix)
    else:
        mat_name1 = epoch_dir + "%s_EEG/%s_%s_repeated_trials.mat" %(subj, subj, save_name_suffix)
        
    mat_dict1 = scipy.io.loadmat(mat_name1)
    im_id_repeat = (mat_dict1['im_id_repeat'][0]).astype(np.int)

    for j in range(nROI):
        tmp_val = np.zeros([len(ROI_ind[j]), n_reg, n_times])
        for j1 in range(n_reg):
            tmpX = X_list[j1][im_id_repeat.astype(np.int)].copy()
            tmpX -= np.mean(tmpX, axis = 0)
            tmp_result = ols_regression(ROI_data[j], tmpX)
            #tmp_val[:,j1,:] = tmp_result['log10p']
            tmp_val[:,j1,:] = tmp_result['Rsq'][:,0:n_times]
        val[i,j] = np.mean(tmp_val,axis = 0)    
        val_normalized[i,j,:] = (tmp_val.transpose([1,2,0]) \
               / np.sum(np.sum(tmp_val, axis = -1),axis = -1)).mean(axis = -1)
        val_normalized_timewise[i,j,:] = (tmp_val.transpose([1,0,2]) \
               / np.sum(tmp_val, axis = 1)).mean(axis = 1)

        
   
diff_val = np.zeros([n_subj,nROI,n_pair,n_times])
for l in range(n_pair):
    diff_val[:,:,l,:] = val[:,:,pairs[l][0]] - val[:,:,pairs[l][1]]
 
mat_dict = dict(val = val, diff_val = diff_val,
                val_normalized = val_normalized,
                val_normalized_timewise = val_normalized_timewise,
                ROI_bihemi_names = ROI_bihemi_names,
                X_name = X_name, pairs = pairs, pair_names = pair_names,
                times = times,
                )
mat_outdir ="/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_regression/dSPM_reg_ROI"        
mat_name =  mat_outdir  + "%s_dSPM_ROI_ols_reg_CCA%d_repeat.mat" %(MEGorEEG[isMEG],n_comp)
scipy.io.savemat(mat_name, mat_dict)   

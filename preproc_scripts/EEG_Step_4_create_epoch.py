# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 21:49:35 2014
ICA Step (2)
We need to inspect the independent components and decide which are bad.
# ===================
July 9, 2014, All subjects except s4 were preprocessed with ICA. 
Most of the blocks had 64 to 68 components, and only 1 ECG and 1-2 EOG components were removed. 
# ===================

@author: ying
"""

import mne
import numpy as np
import scipy, time
import scipy.io
import scipy.stats

import sys
path0 = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/preproc_scripts/"
sys.path.insert(0, path0)
from NEIL_create_epochs import create_epochs, smooth_epochs


tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
filter_dir =  tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"
epoch_dir = tmp_rootdir + "epoch_raw_data/"



flag_switch_PPO10_POO10 = True
if flag_switch_PPO10_POO10:
    fname_suffix = "filter_1_110Hz_notch_PPO10POO10_swapped"
else:
    fname_suffix = "filter_1_110Hz_notch_ica_raw"

print fname_suffix

EOG_list = ['EOG_LO1','EOG_LO2','EOG_IO1','EOG_SO1','EOG_IO2']
ECG_list = ['ECG']
n_eeg_channels = 128

"""
subj_id_seq = [1,2,3,4,5,6,7,8,10,11,12,13,14, 16, 18]    
##subj_list = ['Extra1','Extra2'] 
subj_list = list()
for i in subj_id_seq:
    subj_list.append('Subj%d' % i)
    
n_subj = len(subj_list) 
n_block_per_subject = np.ones(n_subj, dtype = np.int)*6
"""
#subj_id_seq = [1,2,3,4,5,6,7,8,10,11,12,13]    
##subj_list = ['Extra1','Extra2'] 
#subj_list = list()
#for i in subj_id_seq:
#    subj_list.append('Subj%d' % i)  

#subj_list = ["Subj14"]
#subj_list = ["Subj16","Subj18"]


"""
subj_list = ["SubjYY_100", "SubjYY_500"]
n_subj = len(subj_list) 
n_block_per_subject = [3,3]
fname_suffix = "filter_1_110Hz_notch"
"""
subj_list = ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]
n_subj = len(subj_list)
n_block_per_subject = [4,5,6]
flag_switch_PPO10_POO10 = True
if flag_switch_PPO10_POO10:
    fname_suffix = "filter_1_110Hz_notch_PPO10POO10_swapped"
else:
    fname_suffix = "filter_1_110Hz_notch_ica_raw"

print fname_suffix



tmin, tmax = -0.1,1.0
# used to label bad trials, discard the trials where the range > 3std
alpha = 15.0
window_length = 0.05
l_freq = 1.0
h_freq = 110.0
# name of the raw file
# filter_1_110Hz_notch_ica_raw
#====================================================================================

# preprocessing the raw data for the additional subject
if False:
    for i in range(n_subj):    
        subj = subj_list[i]
        if subj in ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]:
            subj +="_EEG"
            print subj
            raw_file_name = ica_dir  + "%s/%s_%s_ica_raw.fif" %(subj,subj,fname_suffix)
            raw = mne.io.Raw(raw_file_name, preload = True)
            print "use frequent counts"
            tmp = raw._data[-1,:].copy()
            counts = np.bincount(tmp[0::100].astype(int))
            baseline = np.argmax(counts)
            raw._data[-1,:] -= baseline
            raw.save(raw_file_name, overwrite = True)
            
        

if False:
    for i in range(n_subj):       
        subj = subj_list[i] + "_EEG"
        print subj
        raw_file_name = ica_dir  + "%s/%s_%s_ica_raw" %(subj,subj,fname_suffix)
        epochs_name =  epoch_dir + "%s/%s_%s_ica_reref" %(subj, subj, fname_suffix)
        create_epochs(raw_file_name, epochs_name, event_id = [100], tmin = tmin, tmax = tmax, 
                      stim_channel = "STI101", channel_number = 128)
        smooth_epochs(epochs_name, window_length = window_length, sfreq = 100.0)
        
if True:
    # fname_suffix is used together to differentialte differently processed epochs
    # smoothed version, 
    #fname_suffix, save_name_suffix = "filter_1_110Hz_notch_ica_reref_smoothed", "1_110Hz_notch_ica_window_%dms" %(window_length*1000)
    # unsmoothed data
    #fname_suffix, save_name_suffix = "filter_1_110Hz_notch_ica_reref",  "1_110Hz_notch_ica"
    #fname_suffix, save_name_suffix = "filter_1_110Hz_notch_ica_reref",  "1_110Hz_notch_ica"
    if flag_switch_PPO10_POO10:
        fname_suffix = "filter_1_110Hz_notch_PPO10POO10_swapped_ica_reref"
        save_name_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped"
    else:
        fname_suffix = "filter_1_110Hz_notch_ica_reref"
        save_name_suffix = "1_110Hz_notch_ica"
    n_eeg_channels = 128
    print fname_suffix
    print save_name_suffix
    sfreq = 100.0
    for i in range(n_subj):
        subj = subj_list[i] + "_EEG"
        n_block = n_block_per_subject[i]
        #epochs_name =  epoch_dir + "%s/%s_%s_ica_reref_smoothed-epo.fif.gz" \
        #              %(subj, subj, fname_suffix)
        epochs_name =  epoch_dir + "%s/%s_%s-epo.fif.gz" \
                     %(subj, subj, fname_suffix)
        epochs = mne.epochs.read_epochs(epochs_name)
        # if the sampling frequency is not 100 Hz, down sample it
        if epochs.info['sfreq'] != sfreq:
            epochs.resample(sfreq = sfreq)
            print "downsampling to 100Hz"
                
        picks_all = mne.pick_channels(epochs.ch_names, epochs.ch_names[0:n_eeg_channels], 
                                      exclude = epochs.info['bads'])
        # but alwayse include the
        epoch_mat = epochs.get_data()[:,picks_all,:]
        times = epochs.times
        n_times = len(times)
        
        n_channels = len(picks_all)
        #======================load the image sequences =================================================
        im_id = np.zeros(0)
        is_repeated = np.zeros(0)
        mat_file_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/Experiment_mat_files/"
    
        for k in range(n_block):
            tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ "%s_block%d_EEG_post_run.mat" %( subj_list[i], k+1))
            # use the indice 0 to 361
            im_id = np.hstack([im_id,tmp_mat['this_im_order'][:,0]])
            is_repeated = np.hstack([is_repeated,tmp_mat['this_is_repeated'][:,0]])
    
        print subj
        print "n_trials"
        print len(im_id)
        print epoch_mat.shape[0]
        
        
        if subj in ["Subj_additional_1_500"]:
            # some triggers in this subject was missing,only match the last 4 blocks 199*2*8 
            n_trials0 = epoch_mat.shape[0]
            epoch_mat = epoch_mat[n_trials0 - len(im_id):n_trials0]
    
        if subj in ['SubjYY_500_EEG']:
            im_id = im_id[0:len(epochs)]
            is_repeated = is_repeated[0:len(epochs)]
    
    
    
        im_id = im_id -1
        print im_id.min()
        epoch_mat_no_repeat = epoch_mat[is_repeated == 0]
        im_id_no_repeat = im_id[is_repeated == 0]
        
        # save the first mat file
        mat_name = epoch_dir + "%s/%s_%s_all_trials.mat" %(subj, subj, save_name_suffix)
        mat_dict = dict(epoch_mat_no_repeat = epoch_mat_no_repeat,
                        im_id_no_repeat = im_id_no_repeat,
                        picks_all = picks_all, times = times)
        scipy.io.savemat(mat_name, mat_dict)
        
        
        #====================== label the bad trials, range  for any channels > alpha std? ==============
        if subj not in ['SubjYY_500_EEG']:
        
            n_trials = epoch_mat_no_repeat.shape[0]
            ranges_each_trial = np.max(epoch_mat_no_repeat, axis = 2) - np.min(epoch_mat_no_repeat, axis = 2)
            ranges_zscore = scipy.stats.zscore(ranges_each_trial, axis = 0)
            bad_trials = np.any(ranges_zscore > alpha, axis = 1)
            print "# bad_trials %d" %bad_trials.sum()
    	
            im_id1 = im_id_no_repeat[bad_trials == 0]
            epoch_mat1 = epoch_mat_no_repeat[bad_trials == 0]
        else:
            im_id1 = im_id_no_repeat
            epoch_mat1 = epoch_mat_no_repeat
        
        n_im = len(np.unique(im_id_no_repeat))
        ave_data = np.zeros([n_im, n_channels, n_times])
        for i in range(n_im):
            if np.sum(im_id1 == i) >0:
                ave_data[i] = np.mean(epoch_mat1[im_id1 == i,:,:], axis = 0)
            else:
                print "im %d not found" %i
        
        print ave_data.shape
        #======== save the avereaged data too ===================
        mat_name = epoch_dir + "%s/%s_%s_ave_alpha%1.1f.mat" %(subj, subj, save_name_suffix, alpha)
        mat_dict = dict(ave_data = ave_data, picks_all = picks_all, times = times,
                        n_im = n_im)
        scipy.io.savemat(mat_name, mat_dict)
    
        
    
## debug
"""
subj = "Subj1_EEG"
save_name_suffix1 = "1_110Hz_notch_ica_PPO10POO10_swapped"
save_name_suffix2 = "1_110Hz_notch_ica"
alpha = 15
mat1 = scipy.io.loadmat(epoch_dir + "%s/%s_%s_ave_alpha%1.1f.mat" %(subj, subj, save_name_suffix1, alpha))
mat2 = scipy.io.loadmat(epoch_dir + "%s/%s_%s_ave_alpha%1.1f.mat" %(subj, subj, save_name_suffix2, alpha))

data1 = mat1['ave_data']
data2 = mat2['ave_data']
diff = data1[:,:,0:100]-data2[:,:,0:100]
print (diff**2).sum(axis = 0).sum(axis = 1) > 1E-10
"""
    

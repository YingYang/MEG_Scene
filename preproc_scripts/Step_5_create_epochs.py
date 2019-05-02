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
import scipy
import scipy.stats
import scipy.io

import sys
path0 = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/preproc_scripts/"
sys.path.insert(0, path0)
from NEIL_create_epochs import create_epochs, smooth_epochs

#===================================================================================
## Subj 1-9
#Subj_list = range(1,10)
#n_runs_per_subject = [6,12,6,12,10,8,12,10,10]
## Subj 10-13
#Subj_list = range(1,14)
#n_runs_per_subject = [6,12,6,12,10,8,12,10,10,10,12,12,12]
#Subj_list = range(14,16)
#n_runs_per_subject = [12,10]
Subj_list = range(16,19)
n_runs_per_subject = [12,12,12]
#====================================================================================

tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
ica_dir = tmp_rootdir + "ica_raw_data/"

fname_suffix = "filter_1_110Hz_notch_ica_raw"
window_length = 0.05

epoch_dir = tmp_rootdir + "epoch_raw_data/"
tmin, tmax = -0.1, 1.0
n_subj = len(Subj_list)

# only do it for the first 7 subjects
if False:
    for i in range(n_subj):
        subj = "Subj%d" %Subj_list[i]
        n_run = n_runs_per_subject[i]
        for j in range(n_run):
            run = "run%d" %(j+1)
            print subj, run
            raw_file_name =  ica_dir + "/%s/%s_%s_%s" %(subj, subj, run, fname_suffix)
            epochs_name =  epoch_dir + "/%s/%s_%s_%s" %(subj, subj, run, "filter_1_110Hz_notch_ica")
            create_epochs(raw_file_name, epochs_name, event_id = [1], tmin = tmin, tmax = tmax,
                          stim_channel = "STI101", channel_number = 306)
            smooth_epochs(epochs_name, window_length = window_length, sfreq = 100.0)

# for Subj9, run 5 (block 3 run1), the first trial was lost, because the trigger was as high as 255, then to 1

    
#==============================================================================
# For each subject, load the epochs, and get all the trials, and save them into a matfile
# Save the corresponding image id, and the index of the bad trials too
# Note Subj 9 run 4 was interupted, it has to be handled differently, only 168 trials were usable
if True:
    # fname_suffix is used together to differentialte differently processed epochs
    # there is an extra "raw" in the file name, but it does not hurt too much, so I did not bother to change it
    # smoothed version, 
    #fname_suffix, save_name_suffix = "filter_1_110Hz_notch_ica_smoothed", "1_100Hz_notch_ica_window_%dms" %(window_length*1000)
    # unsmoothed data
    fname_suffix, save_name_suffix = "filter_1_110Hz_notch_ica",  "1_110Hz_notch_ica"
    print fname_suffix
    print save_name_suffix
    sfreq = 100.0
    for i in range(n_subj):
        subj = "Subj%d" %Subj_list[i]
        n_run = n_runs_per_subject[i]
        print subj
        data_list = list()
        pick_list = list()
        for j in range(n_run):
            run = "run%d" %(j+1)
            epochs_name =  epoch_dir + "%s/%s_%s_%s-epo.fif.gz" %(subj, subj, run, fname_suffix)
            epochs = mne.epochs.read_epochs(epochs_name)
            # interpolate bad trials
            # this epoch will not be saved, so do not worry about it, reset_bads will remove the bad channel info
            if len(epochs.info['bads']) >0:
                epochs.interpolate_bads(reset_bads = True)

            # for unsmoothed data, the sampling rate is not 100Hz, downsample it
            if epochs.info['sfreq'] != sfreq:
                epochs.resample(sfreq = sfreq)  
                print "downsampling to 100 Hz"
                
            print j, subj, run
            print len(epochs)
            picks = mne.pick_types(epochs.info, meg = True)
            pick_list.append(picks)
            print "# channels = %d" % len(picks)
            if subj in ['Subj9'] and run == "run4": 
                # Note Subj 9 run 4 was interupted, it has to be handled differently, 
                # only 168 trials were usable
                tmp_n_trials = 168
                data_list.append(epochs.get_data()[0:tmp_n_trials,0:306,:])
            else:
                data_list.append(epochs.get_data()[:,0:306,:])
            # record the times of the first run, all the remaining runs are the same
            if j == 0:
                times = epochs.times
                
        n_times = len(times)
        picks_all = np.array(pick_list[0])
        for j in range(1,n_run):
            picks_all = np.intersect1d(picks_all, np.array(pick_list[j]))
            
        n_times = len(times)
        n_channels = len(picks_all)
        epoch_mat = data_list[0][:,picks_all,:]
        for j in range(1,n_run):
            epoch_mat = np.vstack([epoch_mat, data_list[j][:,picks_all,:]]) 
        
        #======================load the image sequences =================================================
        im_id = np.zeros(0)
        is_repeated = np.zeros(0)
        mat_file_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/Experiment_mat_files/"
        n_block, n_run_per_block = n_run//2,2
        for k in range(n_block):
            for l in range(n_run_per_block):
                if Subj_list[i] > 3:
                    # Subj18, order was broken: 456-123 swapped, I renamed the files with the suffix rename
                    if Subj_list[i] == 18:
                        print "Subj18", subj
                        tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_MEG_post_run_rename.mat" %(k+1,l+1))
                    else:
                        tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_MEG_post_run.mat" %(k+1,l+1))        
                else:
                    tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_post_run.mat" %(k+1,l+1))
                
                if subj in ['Subj9'] and  k == 1 and l == 1:
                    tmp_n_trial = 168 
                    im_id = np.hstack([im_id, tmp_mat['this_im_order'][0:tmp_n_trial,0],])
                    is_repeated = np.hstack([is_repeated, tmp_mat['this_is_repeated'][0:tmp_n_trial,0],])
                else:
                    # use the indice 0 to 361
                    # There was a huge bug here, I add the new run on top of the old runs!! everything was wrong!!!!
                    im_id = np.hstack([im_id, tmp_mat['this_im_order'][:,0]])
                    is_repeated = np.hstack([is_repeated, tmp_mat['this_is_repeated'][:,0]])
        im_id -= 1 # im_id should start from zero
        print im_id.min()
        epoch_mat_no_repeat = epoch_mat[is_repeated == 0]
        im_id_no_repeat = im_id[is_repeated == 0]
        
        # save the first mat file
        mat_name = epoch_dir + "%s/%s_%s_all_trials.mat" %(subj, subj, save_name_suffix)
        mat_dict = dict(epoch_mat_no_repeat = epoch_mat_no_repeat,
                        im_id_no_repeat = im_id_no_repeat,
                        picks_all = picks_all, times = times)
        scipy.io.savemat(mat_name, mat_dict)



#if False:
#    # 20151214: the scripts was modified: 
#    # bad_trials removed, also, merge the special case of Subj9 to the main loop
#    # testing the old and new ones
#    tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
#    save_name_suffix = "1_50Hz_raw_ica_window_50ms"
#    epoch_dir = tmp_rootdir + "epoch_raw_data/"
#    subj = "Subj1"
#    # new
#    mat_name1 = epoch_dir + "%s/%s_%s_all_trials.mat" %(subj, subj, save_name_suffix)
#    # old
#    mat_name2 = epoch_dir + "%s_%s_all_trials.mat" %(subj, save_name_suffix)
#    mat_dict1 = scipy.io.loadmat(mat_name1)
#    mat_dict2 = scipy.io.loadmat(mat_name2)
#    print np.linalg.norm(mat_dict1['epoch_mat_no_repeat']-mat_dict2['epoch_mat_no_repeat'])
#    # Output :0.0
#    subj = "Subj9"
#    mat_name3 = scipy.io.loadmat(epoch_dir + "%s/%s_%s_all_trials.mat" %(subj, subj, save_name_suffix))
#    print mat_name3['epoch_mat_no_repeat'].shape
#    print mat_name3['im_id_no_repeat'].shape
#    print mat_name3['im_id_no_repeat'].min(),mat_name3['im_id_no_repeat'].max()
#    # 20160203: script modified again, also add an option for of the unsmoothed data
#    # using different fname_suffix, save_name_suffix
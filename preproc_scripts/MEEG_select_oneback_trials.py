# -*- coding: utf-8 -*-
"""

alpha: threshold to drop bad trials.
Compute the average across repitions of images, where the number of reptitions in 
 each image is matched across MEG and EEG. 
After computing the averages, save them in two seperate files under their epoch directory. 
Also compute the empirical lag between the two modalities. 

"""

import numpy as np
import scipy.io
import scipy.stats
import mne


n_im = 362

isMEG = False
MEGorEEG = ['EEG','MEG']
if isMEG:
    subj_list = np.arange(1,19)
    n_runs_per_subject = [6,12,6,12,10,8,12,10,10,10,12,12,12,12,10,12,12,12]
else:
    subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
    n_runs_per_subject = (np.ones(len(subj_list))*6).astype(np.int)
    
n_subj = len(subj_list)


# I need to read the original epoch data 
# The button press response can be found in the matlab mat file or STI102 in MEG
# not recorded in any channels in EEG though. 

meta_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"
MEG_dir = meta_dir + "MEG_DATA/DATA/epoch_raw_data/"
EEG_dir = meta_dir + "EEG_DATA/DATA/epoch_raw_data/"
save_name_suffix = "1_110Hz_notch_ica"
fname_suffix = "filter_1_110Hz_notch_ica"
sfreq = 100.0 



'''
#testing doing epochs only on STI102
subj = "Subj14"
raw_fname = meta_dir + "MEG_DATA/DATA/ica_raw_data/%s/%s_run1_filter_1_110Hz_notch_ica_raw.fif" %(subj, subj)
raw = mne.io.Raw(raw_fname, preload = True)

#mne.set_config('MNE_STIM_CHANNEL','STI102')
baseline = (-0.5, 0)
stim_channel = "STI102"
r_events = mne.find_events(raw, stim_channel,min_duration = 2/raw.info['sfreq'], consecutive = True)
# create epochs based on STI102
r_epochs = mne.Epochs(raw, r_events, tmin = -1.0, tmax = 0.5, baseline = None )

r_evoked = r_epochs[0::].average()
#r_evoked.plot()
 

# plot snr
a = r_epochs.get_data()[:,0:306,:]
snra = a.mean(axis = 0)/a.std(axis = 0)
#plt.imshow(snra)
r_evoked.data[0:306,:] = snra
r_evoked.plot_topo()

# save all trials in which a button press happened
# use MEG/EEG signals to predict
'''




# most RT is within 1000 ms. So I did not redo the epochs. 
epoch_dir = MEG_dir if isMEG else EEG_dir
if isMEG:
    mat_file_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/Experiment_mat_files/"
else:
    mat_file_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/Experiment_mat_files/"
MEG_n_run_per_block = 2       

# for each subject, get the data
for i in range(n_subj):
    subj = "Subj%d" % subj_list[i]
    print subj
    data_list = list()
    if isMEG:
        picks_all = np.arange(0,306)
        n_run = n_runs_per_subject[i]
        for j in range(n_run):
            run = "run%d" %(j+1)
            epochs_name =  epoch_dir + "%s/%s_%s_%s-epo.fif.gz" %(subj, subj, run, fname_suffix)
            epochs = mne.epochs.read_epochs(epochs_name)
            # interpolate bad channels
            # this epoch will not be saved, so do not worry about it, reset_bads will remove the bad channel info
            if len(epochs.info['bads']) >0:
                epochs.interpolate_bads(reset_bads = True)
            # for unsmoothed data, the sampling rate is not 100Hz, downsample it
            if epochs.info['sfreq'] != sfreq:
                epochs.resample(sfreq = sfreq)  
                print "downsampling to 100 Hz"    
            print j, subj, run
            print len(epochs)
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
        epoch_mat = data_list[0][:,picks_all,:]
        for j in range(1,n_run):
            epoch_mat = np.vstack([epoch_mat, data_list[j][:,picks_all,:]]) 
    else:
        # for EEG, I have interpolated the bad channels in the previous step
        epochs_name =  epoch_dir + "%s_EEG/%s_EEG_%s_reref-epo.fif.gz" %(subj, subj, fname_suffix)
        epochs = mne.epochs.read_epochs(epochs_name)
        # if the sampling frequency is not 100 Hz, down sample it
        if epochs.info['sfreq'] != sfreq:
            epochs.resample(sfreq = sfreq)
            print "downsampling to 100Hz"        
        picks_all = mne.pick_channels(epochs.ch_names, epochs.ch_names[0:128], 
                                      exclude = epochs.info['bads'])
        epoch_mat = epochs.get_data()[:,picks_all,:]
        times = epochs.times
        n_times = len(times)
        
    n_channels = len(picks_all)
    
    
    #======================load the image sequences =================================================
    if isMEG:
        n_block = n_runs_per_subject[i]//2
    else:
        n_block = n_runs_per_subject[i]
        
    tmp_RT = np.zeros(0)
    is_repeated = np.zeros(0)
    im_id = np.zeros(0)
    for k in range(n_block):
        if isMEG:
            for l in range(MEG_n_run_per_block):
                if subj_list[i] > 3:
                    # Subj18, order was broken: 456-123 swapped, I renamed the files with the suffix rename
                    if subj_list[i] == 18:
                        print "Subj18", subj
                        tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_MEG_post_run_rename.mat" %(k+1,l+1))
                    else:
                        tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_MEG_post_run.mat" %(k+1,l+1))        
                else:
                    tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_post_run.mat" %(k+1,l+1))
                
                if subj in ['Subj9'] and  k == 1 and l == 1:
                    tmp_n_trial = 168 
                    tmp_RT = np.hstack([tmp_RT, tmp_mat['rt'][0:tmp_n_trial,0],])
                    is_repeated = np.hstack([is_repeated, tmp_mat['this_is_repeated'][0:tmp_n_trial,0],])
                    im_id = np.hstack([im_id,tmp_mat['this_im_order'][:,0]])
                else:
                    # use the indice 0 to 361
                    # There was a huge bug here, I add the new run on top of the old runs!! everything was wrong!!!!
                    tmp_RT = np.hstack([tmp_RT, tmp_mat['rt'][:,0]])
                    is_repeated = np.hstack([is_repeated, tmp_mat['this_is_repeated'][:,0]])
                    im_id = np.hstack([im_id,tmp_mat['this_im_order'][:,0]])
        else:
            tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"_EEG/"+ "%s_block%d_EEG_post_run.mat" %(subj, k+1))
            # use the indice 0 to 361
            tmp_RT = np.hstack([tmp_RT,tmp_mat['rt'][:,0]])
            is_repeated = np.hstack([is_repeated,tmp_mat['this_is_repeated'][:,0]])
            im_id = np.hstack([im_id,tmp_mat['this_im_order'][:,0]])
            
    im_id -= 1 # im_id should start from zero
    epoch_mat_repeat = epoch_mat[is_repeated == 1]
    im_id_repeat = im_id[is_repeated == 1]
    RT_repeat = tmp_RT[is_repeated == 1]
    
    # save the first mat file
    mat_name = epoch_dir + "%s_EEG/%s_%s_repeated_trials.mat" %(subj, subj, save_name_suffix)
    mat_dict = dict(epoch_mat_repeat = epoch_mat_repeat, RT_repeat = RT_repeat,
                    im_id_repeat = im_id_repeat, times = times)
    scipy.io.savemat(mat_name, mat_dict)
    
   
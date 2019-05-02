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
# ===========================================================================
def create_epochs(raw_file_name, epochs_name, event_id = [1], tmin = -0.15, tmax = 1.0,
                  stim_channel = "STI101", channel_number = 306):
    '''
    create epochs
    inputs:
        reject_dict, e.g dict(grad = 4000e-13, mag = 4e-12, eog = 250e-6)
	raw_file_name, full path of the raw file, with no extension
	epochs_name, full path of the epochs to be saved, with no extension
	
	e.g. raw_file_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_DATA/Subj1/filtered/Subj1_run1_tsss_filter_1_110Hz_notch"
	     epochs_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_DATA/Subj1/epochs/Subj1_run1_tsss_filter_1_110Hz_notch"
     event_id = [1,2] except for Subj1
    '''
    print channel_number
    mne.set_config('MNE_STIM_CHANNEL','STI101')
    baseline = (None, 0)
    #h_freq = 50.0
    raw = mne.io.Raw(raw_file_name+".fif",preload = True)
    events = mne.find_events(raw, stim_channel,min_duration = 2/raw.info['sfreq'], consecutive = True)
    print len(events)
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, 
                    proj = True, picks = np.arange(0,channel_number),
                    baseline = baseline, preload = True, reject = None)
    print "epoch length"
    print len(epochs)
    # picks = None, to make sure all channels including the bads are saved.    
    # save epochs and the event info
    epochs.save(epochs_name + "-epo.fif.gz")
    # events_time in seconds
    events_time = [events[j,0]/raw.info['sfreq'] for j in range(len(events))  \
                                 if events[j,2] in event_id ]                   
    event_m_file =  epochs_name+"_events.mat"
    scipy.io.savemat(event_m_file, dict(events_time = events_time))
    del(raw)
    del(epochs)
    del(events)
    return 0
 
# seperate smooth and detrend into different functions, 
# incase I needed different smoothing window for them
# ===========================================================================

def smooth_epochs(epochs_name, window_length = 0.05, sfreq = 100.0):
    '''
    create smoothed and downsampled epochs
    inputs:
        epochs_name, full path of the epochs to be saved, with no extension
	e.g. epochs_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_DATA/Subj1/epochs/Subj1_run1_tsss_filter_1_110Hz_notch"
    '''
    epochs = mne.read_epochs(epochs_name + "-epo.fif.gz")
    smoothed_epochs = epochs.copy()
    epochs_data = epochs.get_data()
    # create a smooth filter, using the hanning window
    window_len =((np.int(window_length*epochs.info['sfreq'])//2)*2+1)
    smooth_filter = np.hanning(window_len)
    smooth_filter /= smooth_filter.sum()
    # smooth and detrend
    smoothed_epochs_data = epochs_data.copy()
    for j in range(epochs_data.shape[0]):
        for k in range(epochs_data.shape[1]):
            tmp_smooth = np.convolve(smooth_filter, epochs_data[j,k,:], mode = 'same')
            smoothed_epochs_data[j,k,:] = tmp_smooth
    # put the data back        
    smoothed_epochs._data = smoothed_epochs_data
    # resample the smoothed epochs        
    smoothed_epochs.resample(sfreq = sfreq)
    smoothed_epochs_name = epochs_name+ "_smoothed" + "-epo.fif.gz"
    smoothed_epochs.save(smoothed_epochs_name)
    del(epochs, smoothed_epochs)
    return 0

#============================================================================
def detrend_epochs(epochs_name, window_length = 0.05, sfreq = 1000.0):
    '''
    create detrended epochs for power analysis
    inputs:
        epochs_name, full path of the epochs to be saved, with no extension
	e.g. epochs_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_DATA/Subj1/epochs/Subj1_run1_tsss_filter_1_110Hz_notch"
    '''
    epochs = mne.read_epochs(epochs_name + "-epo.fif.gz")
    detrended_epochs = epochs.copy()
    epochs_data = epochs.get_data()
    # create a smooth filter, using the hanning window
    window_len =((np.int(window_length*epochs.info['sfreq'])//2)*2+1)
    smooth_filter = np.hanning(window_len)
    smooth_filter /= smooth_filter.sum()
    # smooth and detrend
    detrended_epochs_data = epochs_data.copy()
    for j in range(epochs_data.shape[0]):
        for k in range(epochs_data.shape[1]):
            tmp_smooth = np.convolve(smooth_filter, epochs_data[j,k,:], mode = 'same')
            detrended_epochs_data[j,k,:] = epochs_data[j,k,:]- tmp_smooth
    # put the data back        
    detrended_epochs._data = detrended_epochs_data
    detrended_epochs_name = epochs_name + "_detrended"+ "-epo.fif.gz"
    detrended_epochs.save(detrended_epochs_name)
    del(epochs, detrended_epochs)
    return 0
          

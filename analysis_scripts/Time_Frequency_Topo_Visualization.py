# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import mne
import numpy as np
import scipy, time
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_morlet

# ====================for future, these should be written in a text file============
Subj_list = range(1,14)
n_runs_per_subject = [6,12,6,12,10,8,12,10,10,10,12,12,12]
#====================================================================================



# use the raw data
tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
ica_dir = tmp_rootdir + "ica_raw_data/"
fname_suffix = "filter_1_110Hz_notch_ica"
epoch_dir = tmp_rootdir + "epoch_raw_data/"
tmin, tmax = -0.1, 1.0
n_subj = len(Subj_list)

# i th subject, j th epoch
# run it on the raw epochs?
i = 3
j = 0


# grand average
ave_data = np.zeros([306, 1101])

for i in range(len(Subj_list)):
    subj = "Subj%d" %Subj_list[i]
    n_run = n_runs_per_subject[i]
    
    psd_mean = np.zeros([306,26])
    offset = 0.04
    wsize = 160
    tstep = 40
    tmp_stft = np.zeros([81,28])
    for j in range(n_run):
        run = "run%d" %(j+1)
        print subj, run
        
        epochs_name =  epoch_dir + "/%s/%s_%s_%s-epo.fif.gz" %(subj, subj, run, fname_suffix)
        epochs = mne.read_epochs(epochs_name)
        
        ave_data += epochs.average().data
        
    
        freqs = np.arange(5, 100, 2)  # define frequencies of interest
        n_cycles = np.round((freqs / 3.)+1 ) # different number of cycle per frequency
        
        fmin = 0.0
        fmax = 100.0
        psds, freqs = mne.time_frequency.compute_epochs_psd(epochs, fmin = fmin, fmax = fmax)
    
        psd_mean += psds.mean(axis = 0)
        
        
        for r in range(len(epochs)):
            tmp = mne.time_frequency.stft(epochs[r].get_data()[0], wsize, tstep )
            tmp_stft += np.mean(np.abs(tmp), axis = 0)
        
    stft_freqs = mne.time_frequency.stftfreq(wsize, epochs.info['sfreq'])
    tmp_stft = tmp_stft[stft_freqs < 100.0,:]
    stft_freqs = stft_freqs[stft_freqs<100.0]
    stft_times = np.arange(epochs.tmin, epochs.tmax, tstep/epochs.info['sfreq'])-offset
        
        
    tmp_stft = (tmp_stft.T -np.mean(tmp_stft[:, stft_times < 0], axis = 1)).T
    plt.imshow(tmp_stft, interpolation = "none", 
              extent = [stft_times[0], stft_times[-1], stft_freqs[0], stft_freqs[-1]],
                origin = "lower", aspect = "auto")
    plt.colorbar()
        
        
        
        
        
    
    plt.imshow(psd_mean, aspect = "auto", interpolation = "None",
               extent = [fmin, fmax, 0, 306])
    plt.plot(freqs, np.mean(psds, axis = 0).mean(axis = 0))
       
    
    
    
    
    
    
    
    
    if False:
        # I need to update my toolbox in order to use tfr_morlet
        
        power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=True, decim=3, n_jobs=2)
    
        # Baseline correction can be applied to power or done in plots
        # To illustrate the baseline correction in plots the next line is commented
        # power.apply_baseline(baseline=(-0.5, 0), mode='logratio')
        
        #power
        power.plot_topo(baseline=(-0.1, 0), mode='logratio', title='Average power')
        
        fig, axis = plt.subplots(1, 2, figsize=(7, 4))
        power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
                           baseline=(-0.1, 0), mode='logratio', axes=axis[0],
                           title='Alpha')
        power.plot_topomap(ch_type='grad', tmin=0.5, tmax=1.5, fmin=13, fmax=25,
                           baseline=(-0.1, 0), mode='logratio', axes=axis[1],
                           title='Beta')
        mne.viz.tight_layout()
        
        #ITC
        itc.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds')
        
    


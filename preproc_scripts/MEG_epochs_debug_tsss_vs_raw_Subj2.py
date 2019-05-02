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
import matplotlib.pyplot as plt
import scipy

run = "run5"

rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
epochs_name1 = rootdir + "Subj_1_3_not_used/Subj2/epochs/Subj2_%s_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz" %run
epochs_name2 = rootdir + "epoch_raw_data/Subj2/Subj2_%s_filter_1_50Hz_raw_ica_raw_smoothed-epo.fif.gz"  %run
epochs1 = mne.read_epochs(epochs_name1)
epochs2 = mne.read_epochs(epochs_name2)


ch_names1 =  epochs1.ch_names
ch_names2 =  epochs2.ch_names


ch_name = ['MEG2221']

pick1 = mne.pick_channels(epochs1.info['ch_names'], include = ch_name)
pick2 = mne.pick_channels(epochs2.info['ch_names'], include = ch_name)

pick1 = np.arange(0,306)
pick2 = np.arange(0,306)

tmin,tmax = -0.1,1.0
time_ind1 = np.all( np.vstack([epochs1.times> tmin, epochs1.times< tmax ]), axis = 0)
time_ind2 = np.all( np.vstack([epochs2.times> tmin, epochs2.times< tmax ]), axis = 0)
data1 = epochs1._data[:, pick1,:]
data1 = data1[:,:,time_ind1]
data2 = epochs2._data[:, pick2,:]
data2 = data2[:,:,time_ind2]

data10 = data1 - np.mean(data1, axis = 0)
data20 = data2 - np.mean(data2, axis = 0)

plt.figure()
plt.plot(np.mean(data1,axis = 0).ravel(), np.mean(data2,axis = 0).ravel(), '.')

plt.figure()
plt.plot(data10.ravel(), data20.ravel(), '.')
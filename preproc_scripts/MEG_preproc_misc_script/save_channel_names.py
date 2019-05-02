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

#===================================================================================
Subj_list = range(1,10)
n_runs_per_subject = [6,12,6,12,10,8,12,10,10]
#====================================================================================

tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
ica_dir = tmp_rootdir + "ica_raw_data/"
fname_suffix = "filter_1_50Hz_raw_ica_raw"
window_length = 0.05
save_name_suffix = "1_50Hz_raw_ica_window_%dms" %(window_length*1000)
epoch_dir = tmp_rootdir + "epoch_raw_data/"
tmin, tmax = -0.1, 1.0
n_subj = len(Subj_list)

subj = "Subj1"
run = "run1"
epochs_name =  epoch_dir + "/%s/%s_%s_%s-epo.fif.gz" %(subj, subj, run, fname_suffix)
epochs = mne.read_epochs(epochs_name)

ch_names = epochs.info['ch_names']


ch_name_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/preproc_scripts/MEG_preproc_misc_script/ch_names.mat"
scipy.io.savemat(ch_name_fname, dict(ch_names = ch_names))

ch_name_fname1 = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/preproc_scripts/MEG_preproc_misc_script/ch_names.txt"
text_file = open(ch_name_fname1, "w")
for i in range(len(ch_names)):
    text_file.write(ch_names[i] + "\n")
text_file.close()
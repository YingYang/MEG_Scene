# -*- coding: utf-8 -*-
"""
# Subj15, after run3 was too noisy. So I used 1-50Hz filtering to compute the ICA, instead of 1-110Hz
"""

import mne
import numpy as np
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import time
import scipy.io 


# ====================for future, these should be written in a text file============
#====================================================================================



tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
filter_dir = tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"


l_freq = 1.0
h_freq = 110.0
notch_freq = [60.0,120.0]
fname_suffix = "filter_%d_%dHz_notch_raw" %(l_freq, h_freq)
decim = 10
var_explained_all = 1-1E-4
var_explained1 = 1-1E-7

n_runs = 10
# when decim = 50, many runs had no EOG detected, change it to 10 helped. 
print decim
if True:
    subj = "Subj%d" %15
    print subj
    for j in range(n_runs):
    # only for subject 2
    #for j in range(9,12):
        run = "run%d" %(j+1)
        fif_name = filter_dir + "%s/%s_%s_%s.fif" %( subj, subj, run, fname_suffix)
        ica_out_fname = ica_dir + "%s/%s_%s_%s_ica_obj-ica.fif" %(subj, subj, run, fname_suffix)
        raw = mne.io.Raw(fif_name,preload = True)
        # filter the raw data to 1-50Hz
        raw.filter(l_freq, h_freq = 50.0)
    
        # ============= ICA ================================================
        var_explained = var_explained_all
        print subj, j, var_explained
        ica = ICA(n_components=var_explained, max_pca_components=None)
        # which sensors to use
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,ecg = False,
                   stim=False, exclude='bads')
        # compute the ica components ( slow ), no rejection was applied
        tmp_time = time.time()
        # decim : downsample the data a bit to compute the ica, reduce computational burden.
        ica.fit(raw, picks=picks, decim = decim) 
        print time.time()-tmp_time
    	
        #===== EOG ======
        # create EOG epochs to improve detection by correlation
        picks = mne.pick_types(raw.info, meg=True, eog=True)
        eog_epochs = create_eog_epochs(raw, picks=picks)
        #eog_inds: indices of ica components that are correlated with EOG channels
        #eog_scores: correlation scores of the components
        eog_inds, eog_scores = ica.find_bads_eog(eog_epochs)   
        #====== ECG ======
        ecg_epochs = create_ecg_epochs(raw, picks=picks)
        ecg_inds, ecg_scores = ica.find_bads_ecg(ecg_epochs)   
        # =================== save the results =====================================
        ica.save(ica_out_fname)
        mat_name = ica_dir + "%s/%s_%s_%s_ECG_EOG.mat" %(subj, subj, run, fname_suffix)
        mat_dict = dict(eog_inds = eog_inds, eog_scores = eog_scores,
        		ecg_inds = ecg_inds, ecg_scores = ecg_scores)
        scipy.io.savemat(mat_name, mat_dict)
        del(raw)
        del(ica)
        del(mat_dict)
        print "%s %s finished" % (subj, run)
            

#20160220:
# decim = 50
# In the ICA computed on 20160218: ICA object for Subj4 has only 28 components, 
# which was totally off.  Recompute the ICA on 20160220 
# i,j = 3,0     # Subj4, run1,
# ica = ICA(n_components=1-1E-7, max_pca_components=None)  # add more ICA components
# "Subj4 run1 finished"

#20160221
# In the ICA computed on 20160218: ICA object for Subj8 run1,2 has no EOG, indicating
# some error. Re-running it with decim = 10, there is an EOG component. 
# I re-run it for Subj1-13 together, with decim = 10


    

   

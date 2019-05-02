# -*- coding: utf-8 -*-
"""
Subj4, run1 has only 30 IC components, which is totally wrong. 
Cut the data after 408 seconds, the channels went crazy after it. 
Rename the original filtered file as *_old.fif, then save the new one.  
"""

import mne
import numpy as np
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
import time
import scipy.io 


# ====================for future, these should be written in a text file============
Subj_list = range(1,10)
n_runs_per_subject = [6,12,6,12,10,8,12,10,10]
#====================================================================================



tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
filter_dir = tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"


n_subj = len(Subj_list)
l_freq = 1
h_freq = 50
fname_suffix = "filter_%d_%dHz_raw" %(l_freq, h_freq)
#for i in range(2,n_subj):
if True:
    i = 7
    subj = "Subj%d" %Subj_list[i]
    print subj
    #for j in range(n_runs_per_subject[i]):
    # only for subject 2
    #for j in range(9,12):
    if True:
        j = 9
        run = "run%d" %(j+1)
        # this can only be run once!!!
        fif_name = filter_dir + "%s/%s_%s_%s.fif" %( subj, subj, run, fname_suffix)
        ica_out_fname = ica_dir + "%s/%s_%s_%s_ica_obj-ica.fif" %(subj, subj, run, fname_suffix)
        raw = mne.io.Raw(fif_name,preload = True)

    
        # ============= ICA ================================================
        ica = ICA(n_components=0.999, max_pca_components=None)
        # which sensors to use
        picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False,ecg = False,
                   stim=False, exclude='bads')
        # compute the ica components ( slow ), no rejection was applied
        tmp_time = time.time()
        # decim : downsample the data a bit to compute the ica, reduce computational burden.
        ica.fit(raw, picks=picks, decim=5) 
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
            
               
    

   

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
import scipy.io, time


# ====================for future, these should be written in a text file============
#Subj_list = range(1,14)
#n_runs_per_subject = [6,12,6,12,10,8,12,10,10,10,12,12,12]
#Subj_list = range(14,16)
#n_runs_per_subject = [12,10]
Subj_list = range(16,19)
n_runs_per_subject = [12,12,12]
#====================================================================================

tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
filter_dir =  tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"

l_freq = 1.0
h_freq = 110.0
notch_freq = [60.0,120.0]
fname_suffix = "filter_%d_%dHz_notch_raw" %(l_freq, h_freq)

n_subj = len(Subj_list)

if True:
    for i in range(n_subj):
        subj = "Subj%d" %Subj_list[i]
        run_list = list()
        n_run = n_runs_per_subject[i]
        for j in range(n_run):
            run_list.append("run%d" %(j+1))
    
        for j in range(n_run):
            run = run_list[j]
            print subj, run
            fif_name = filter_dir  + "%s/%s_%s_%s.fif" %(subj,subj,run,fname_suffix)
            raw = mne.io.Raw(fif_name,preload = True)
            # ============= load ICA =========================================================
            ica_name = ica_dir  + "%s/%s_%s_%s_ica_obj-ica.fif" %(subj,subj,run,fname_suffix)
            ica = mne.preprocessing.read_ica(ica_name)
            new_mat_name = ica_dir  + "%s/%s_%s_%s_ECG_EOG_manual_check.mat" %(subj,subj,run,fname_suffix)
            new_mat = scipy.io.loadmat(new_mat_name)       
            if len(new_mat['new_eog_inds'])>0:
                new_eog_inds = new_mat['new_eog_inds'][0].astype(np.int)
            else:
                new_eog_inds = []
            if len(new_mat['new_ecg_inds'])>0:
                new_ecg_inds = new_mat['new_ecg_inds'][0].astype(np.int)
            else:
                new_ecg_inds = []
            
            exclude = (np.union1d(new_eog_inds, new_ecg_inds)).tolist()
            print exclude
            ica.exclude = exclude
            
            tmp_t = time.time()
            raw_after_ica = ica.apply(raw,exclude = ica.exclude)  
            ica_raw_name = ica_dir  + "%s/%s_%s_filter_%d_%dHz_notch_ica_raw.fif" %(subj,subj,run,l_freq,h_freq)
            raw_after_ica.save(ica_raw_name, overwrite = True)  
            print time.time()-tmp_t
            # =============== clean the objects to release memory ======================
            del(raw)
            del(ica)
            del(raw_after_ica)

    
# 20160229
# The ICA removal for Subj8 was bad. I reused the ICA object, and components obtained for 1-50Hz, for all Runs
if False:
    fname_suffix1 = "filter_1_50Hz_raw"
    i= 7
    subj = "Subj%d" %Subj_list[i]
    run_list = list()
    n_run = n_runs_per_subject[i]
    for j in range(n_run):
        run_list.append("run%d" %(j+1))
    
    for j in range(n_run):
        run = run_list[j]
        print subj, run
        fif_name = filter_dir  + "%s/%s_%s_%s.fif" %(subj,subj,run,fname_suffix)
        raw = mne.io.Raw(fif_name,preload = True)
        # ============= load ICA =========================================================
        ica_name = ica_dir  + "%s/%s_%s_%s_ica_obj-ica.fif" %(subj,subj,run,fname_suffix1)
        ica = mne.preprocessing.read_ica(ica_name)
        new_mat_name = ica_dir  + "%s/%s_%s_%s_ECG_EOG_manual_check.mat" %(subj,subj,run,fname_suffix1)
        new_mat = scipy.io.loadmat(new_mat_name)       
        if len(new_mat['new_eog_inds'])>0:
            new_eog_inds = new_mat['new_eog_inds'][0].astype(np.int)
        else:
            new_eog_inds = []
        if len(new_mat['new_ecg_inds'])>0:
            new_ecg_inds = new_mat['new_ecg_inds'][0].astype(np.int)
        else:
            new_ecg_inds = []
        
        exclude = (np.union1d(new_eog_inds, new_ecg_inds)).tolist()
        print exclude
        ica.exclude = exclude
        
        tmp_t = time.time()
        raw_after_ica = ica.apply(raw,exclude = ica.exclude)  
        ica_raw_name = ica_dir  + "%s/%s_%s_filter_%d_%dHz_notch_ica_raw.fif" %(subj,subj,run,l_freq,h_freq)
        raw_after_ica.save(ica_raw_name, overwrite = True)  
        print time.time()-tmp_t
        # =============== clean the objects to release memory ======================
        del(raw)
        del(ica)
        del(raw_after_ica)

                       

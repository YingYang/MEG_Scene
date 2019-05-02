# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 21:49:35 2014
bandpass and notch filter both the empty data and the raw data
@author: ying
"""


import mne
import numpy as np


# ====================for future, these should be written in a text file============
## Subj 1-9
#Subj_list = range(1,10)
#n_runs_per_subject = [6,12,6,12,10,8,12,10,10]
## Subj 10-13
## Subj 1-13
#Subj_list = range(1,14)
#n_runs_per_subject = [6,12,6,12,10,8,12,10,10,10,12,12,12]
#Subj_list = range(14,16)
#n_runs_per_subject = [12,10]
Subj_list = range(16,19)
n_runs_per_subject = [12,12,12]
#====================================================================================
tmp_root_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/raw_data/"
tmp_out_dir ="/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/filtered_raw_data/"
# ===========================================================================
# No notch filter, 1-50 bandpass
#
# rejection for SSP in the empty room data
reject_dict = dict(grad=2000e-13, # T / m (gradiometers)
              mag=3e-12, # T (magnetometers)
              )
# MNE manual  projgradrej 2000fT/cm  projmagrej 3000 fT
# but it also says these values are system dependent              

n_subj = len(Subj_list)
l_freq = 1.0
h_freq = 110.0
notch_freq = [60.0,120.0]
fname_suffix = "filter_%d_%dHz_notch_raw" %(l_freq, h_freq)
for i in range(n_subj):
    subj = "Subj%d" %Subj_list[i]
    # load the channels to remove
    raw_file_dir = tmp_root_dir  + subj + "/"
    run_names = ["emptyroom"]
    for j in range(n_runs_per_subject[i]):
        run_names.append("run%d" %(j+1))
    
    # first run is always empty room
    j = 0
    print run_names[j]
    raw_name = raw_file_dir + "Bad_Channel_Marked/NEIL_%s_%s_raw.fif" %(subj, run_names[j])
    er_raw = mne.io.Raw(raw_name, preload = True)
    er_raw.notch_filter(notch_freq)
    er_raw.filter(l_freq, h_freq) 
    er_raw.add_proj([], remove_existing = True)
    # the SSP for the emptyroom data is set to zero
    projs = mne.proj.compute_proj_raw(er_raw, duration = 2.0, n_grad = 2, n_mag = 3,
                                          n_jobs = 2, reject = reject_dict)
    proj_fname = tmp_out_dir  +  subj  +  "/" + subj+ "_" + "emptyroom" +"_"+ fname_suffix +"_SSP.fif"
    mne.proj.write_proj(proj_fname,projs)
    out_name = tmp_out_dir  + subj + "/%s_%s_%s.fif" %(subj, run_names[j], fname_suffix)
    er_raw.save(out_name, overwrite = True)
    del(er_raw)
    for j in range(1,len(run_names)):
        print run_names[j]
        raw_name = raw_file_dir + "Bad_Channel_Marked/NEIL_%s_%s_raw.fif" %(subj, run_names[j])
        raw = mne.io.Raw(raw_name, preload = True)
        raw.filter(l_freq, h_freq)
        raw.notch_filter(notch_freq)
        raw.add_proj(projs,remove_existing = True)
        out_name = tmp_out_dir  + subj + "/%s_%s_%s.fif" %(subj, run_names[j], fname_suffix)
        print out_name
        raw.save(out_name, overwrite = True)
        del(raw)
    
    

               

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 21:49:35 2014
bandpass and notch filter both the empty data and the raw data
@author: ying
"""


import mne
import numpy as np
mne.set_log_level('WARNING')


tmp_root_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/raw_data/"


# ====================for future, these should be written in a text file============
## Subj 1-9
#Subj_list = range(1,10)
#n_runs_per_subject = [6,12,6,12,10,8,12,10,10]
## Subj 10-13
#Subj_list = range(10,14)
#n_runs_per_subject = [10,12,12,12]
#Subj_list = range(14,16)
#n_runs_per_subject = [12,10]
Subj_list = range(16,19)
n_runs_per_subject = [12,12,12]
#====================================================================================
n_subj = len(Subj_list)

for i in range(n_subj):
    subj = "Subj%d" %Subj_list[i]
    # load the channels to remove
    raw_file_dir = tmp_root_dir  + subj + "/"
    bad_channel_list_name = raw_file_dir + "NEIL_%s_Bad_Channel_List_50Hz.txt" %subj
    # ignore comments #, the first line is always the emtpy room data
    bad_channel_list = list()
    f = open(bad_channel_list_name)
    for line in f:
        if line[0]!= "#":
            print line.split()
            bad_channel_list.append(line.split())
    f.close()
    
    for k in range(len(bad_channel_list)):
        for l in range(len(bad_channel_list[k])):
            bad_channel_list[k][l] = "MEG" + bad_channel_list[k][l] 
    
    #run_names
    run_names = ["emptyroom"]
    for j in range(n_runs_per_subject[i]):
        run_names.append("run%d" %(j+1))
    
    for j in range(len(run_names)):
        raw_name = raw_file_dir + "intact/NEIL_%s_%s_raw.fif" %(subj, run_names[j])
        raw = mne.io.Raw(raw_name)
        raw.info['bads'] = bad_channel_list[j]
        print i, j, raw.info['bads']
        # change the line frequency too
        raw.info['line_freq'] = 60.0
        print raw.info['line_freq']
        # same the same file, overwrite
        raw_name1 = raw_file_dir + "Bad_Channel_Marked/NEIL_%s_%s_raw.fif" %(subj, run_names[j])
        raw.save(raw_name1, overwrite = True)
        
    del(raw, bad_channel_list, raw_file_dir, run_names)
        
   
    




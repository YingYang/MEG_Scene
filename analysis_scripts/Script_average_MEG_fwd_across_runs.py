# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import mne 
fwd_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/fwd/"

#subj_list = range(1,9)
#n_runs_per_subj = [6,12,6,12,10,8,12,10]
#subj_list = range(9,14)
#n_runs_per_subj = [10,10,12,12,12]

subj_list = range(14,19)
n_runs_per_subj = [12,10,12,12,12]
n_subj = len(subj_list)

for i in range(n_subj):
    fwd_list = list()
    subj_id = subj_list[i]
    subj = "Subj%d" %subj_id
    print subj
    print n_runs_per_subj[i]
    for j in range(n_runs_per_subj[i]):
        if subj_id >= 14:
            tmp_fwd_path = fwd_dir + "%s/%s_oct-6_run%d-fwd.fif" %(subj, subj,j+1)
        else:
            tmp_fwd_path = fwd_dir + "%s/%s_run%d-fwd.fif" %(subj, subj,j+1)
        print tmp_fwd_path
        fwd_list.append(mne.read_forward_solution(tmp_fwd_path))
    print j
    
    fwd_ave = mne.average_forward_solutions(fwd_list)
    fwd_out_path = fwd_dir + "%s/%s_ave-fwd.fif" %(subj, subj)
    mne.write_forward_solution(fwd_out_path, fwd_ave, overwrite = True)
    

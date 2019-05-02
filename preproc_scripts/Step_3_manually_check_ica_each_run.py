# -*- coding: utf-8 -*-
"""
Manual scripts, to save the ica components to exclude. 

@author: ying
"""

import mne
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import scipy.io
import scipy.stats
import time



# ====================for future, these should be written in a text file============
## Subj 1-9
#Subj_list = range(1,10)
#n_runs_per_subject = [6,12,6,12,10,8,12,10,10]
## Subj 10-13
Subj_list = range(1,19)
n_runs_per_subject = [6,12,6,12,10,8,12,10,10,10,12,12,12,12,10,12,12,12]
#====================================================================================

tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/"
filter_dir =  tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"

i = 17
subj = "Subj%d" %Subj_list[i]
run_list = list()
n_run = n_runs_per_subject[i]
for j in range(n_run):
    run_list.append("run%d" %(j+1))

l_freq = 1.0
h_freq = 110.0
notch_freq = [60.0,120.0]
fname_suffix = "filter_%d_%dHz_notch_raw" %(l_freq, h_freq)
fig_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/preproc_scripts/preproc_ica_MEG_figs/raw_filter/%s/" % subj 


    # the following process needs to be run individually for each run
    #================setting which participant and which run ==================
if True:
    #for j in range(len(run_list)):
    plt.close('all')
    j = 7
    run = run_list[j]
    print subj, run
    
    if True:
        fif_name = filter_dir  + "%s/%s_%s_%s.fif" %(subj,subj,run,fname_suffix)
        raw = mne.io.Raw(fif_name,preload = True)
        
        # change after upgrading to MNE0.10, ica.plot will plot the entire raw, which takes 5 min
        # so we crop the raw data to reduce the plotting time
        start, stop = 0.0,300.0
        raw = raw.crop(tmin = start,tmax = stop) 
        
        # ==== debug, this is to make sure the projection saved in the raw data is correct
        #layouts = mne.find_layout(raw.info, "meg") 
        #mne.viz.plot_projs_topomap(raw.info['projs'], layout=layouts)
        #
        #plt.figure()
        #proj_name =filter_path  + subj + "_" + "emptyroom" +"_"+ fname_suffix +"_SSP.fif"
        #projs = mne.proj.read_proj(proj_name)
        #mne.viz.plot_projs_topomap(projs, layout=layouts)
        
        #raw.add_proj(projs,remove_existing = True) # this should already be added
        # ============= load ICA =========================================================
        ica_name = ica_dir  + "%s/%s_%s_%s_ica_obj-ica.fif" %(subj,subj,run,fname_suffix)
        ica = mne.preprocessing.read_ica(ica_name)
    mat_name = ica_dir  + "%s/%s_%s_%s_ECG_EOG.mat" %(subj,subj,run,fname_suffix)
    EXG_dict = scipy.io.loadmat(mat_name) 
    eog_scores = EXG_dict['eog_scores'][0:2,:]
    ecg_scores = EXG_dict['ecg_scores'][0]
    
    # Subj4, run1, raw, the EOG/ECG scores were totally off, 20150908, ICA obtained on 20150905
    if len(EXG_dict['eog_inds'])>0:
        eog_inds = EXG_dict['eog_inds'][0]
    else: 
        # if no automatic EOG was detected, use the first 10 ICs that are mostly
        # correlated with EOG1
        print "empty EOG inds"
        eog_inds = np.argsort(np.abs(eog_scores[0]))[-1:-10:-1]
    
    if len(EXG_dict['ecg_inds'])>0:
        ecg_inds = EXG_dict['ecg_inds'][0]
    else: 
        # if no automatic EOG was detected, use the first 10 ICs that are mostly
        # correlated with EOG1
        print "empty ECG inds"
        ecg_inds = np.argsort(np.abs(ecg_scores))[-1:-10:-1]
    
    
    total_components = ica.n_components_
    # =================plots ===================================================
    print eog_inds
    print ecg_inds
    
    #Subj5 Run9, ecg_inds = [90, 223, 122]
    #Subj9 Run3, eog_inds = [245]
    
    if True:
        # there are too many components to visualize, I only select from the automatically detected ones
        ica.plot_scores(eog_scores, exclude=eog_inds)
        tmp_fig_name = fig_dir  + "%s_%s_%s_EOG_score.png" %(subj,run,fname_suffix)
        plt.savefig(tmp_fig_name)
        plt.close()
        
        ica.plot_scores(ecg_scores, exclude=ecg_inds)
        tmp_fig_name = fig_dir  + "%s_%s_%s_ECG_score.png" %(subj,run,fname_suffix)
        plt.savefig(tmp_fig_name)
        plt.close()
    if True:
        # these numbers must be float
        
        t0 = time.time()
        fig1 = ica.plot_sources(raw,picks = eog_inds)
        tmp_fig_name = fig_dir  + "%s_%s_%s_EOG_ts.png" %(subj,run,fname_suffix)
        fig1.savefig(tmp_fig_name)

        fig2 = ica.plot_sources(raw, picks = ecg_inds )
        tmp_fig_name = fig_dir  + "%s_%s_%s_ECG_ts.png" %(subj,run,fname_suffix)
        fig2.savefig(tmp_fig_name)
        #plt.close()
        print time.time()-t0
        
        # changes after upgrading to MNE 0.10
        # press "-" to reduce scales
        # then re-run the following        
        tmp_fig_name = fig_dir  + "%s_%s_%s_EOG_ts.png" %(subj,run,fname_suffix)
        fig1.savefig(tmp_fig_name)
        tmp_fig_name = fig_dir  + "%s_%s_%s_ECG_ts.png" %(subj,run,fname_suffix)
        fig2.savefig(tmp_fig_name)
        
    
    # ================ make manual selections =====================================
    #new_eog_inds = eog_inds[1:2]
    if True:
        # Subj6 Run2        new_eog_inds = eog_inds[0:8]
        # Subj6 Run3        new_eog_inds = eog_inds[[0,1,2,3,4,5,6,7,9,10,12]]
        # Subj6 Run4        new_eog_inds = eog_inds[0:10]
        # Subj6 Run5        new_eog_inds = eog_inds[[0,1,2,3,4,5,6,7,10,11]]
        # Subj6 Run6        new_eog_inds = eog_inds[0:-1]
        # Subj6 Run8        new_eog_inds = eog_inds[0:-2]
        #new_eog_inds = eog_inds[np.argsort(-np.abs(eog_scores[0,eog_inds]))[0:-2]]
        new_eog_inds = eog_inds[0:1]
        #new_eog_inds = eog_inds[0:2]
        #new_eog_inds  = [4,36,152] # Subj17, run 6
        #new_ecg_inds = ecg_inds[0:1]
        new_ecg_inds = ecg_inds[0:2]
        
        print new_eog_inds, new_ecg_inds
        new_mat = dict(new_eog_inds = new_eog_inds, new_ecg_inds = new_ecg_inds)
        new_mat_name = ica_dir  + "%s/%s_%s_%s_ECG_EOG_manual_check.mat" %(subj,subj,run,fname_suffix)
        print new_mat_name
        scipy.io.savemat(new_mat_name, new_mat)
    
    
    if False:
        del(ica)
        del(raw)
        

if False:
    # print the eog and ecg inds at the end
    for j in range(len(run_list)):
        run = run_list[j]
        new_mat_name = ica_dir  + "%s/%s_%s_%s_ECG_EOG_manual_check.mat" %(subj,subj,run,fname_suffix)
        tmp_dict = scipy.io.loadmat(new_mat_name)
        print run, tmp_dict['new_eog_inds'], tmp_dict['new_ecg_inds']
        print run, np.sort(tmp_dict['new_eog_inds']), np.sort(tmp_dict['new_ecg_inds'])
        del(tmp_dict)
    
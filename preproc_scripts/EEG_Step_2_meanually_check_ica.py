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



#====================================================================================
tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
filter_dir =  tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"

EOG_list = ['EOG_LO1','EOG_LO2','EOG_IO1','EOG_SO1','EOG_IO2']
ECG_list = ['ECG']


# ====================for future, these should be written in a text file============
#subj_id_seq = [1,2,3,4,5,6,7,8,10,11,12,13]    
##subj_list = ['Extra1','Extra2'] 
#subj_list = list()
#for i in subj_id_seq:
#    subj_list.append('Subj%d' % i)
    
    
#subj_list = ["Subj14"] 
#subj_list = ["Subj16", "Subj18"]  
#subj_list = ['SubjYY_100', 'SubjYY_500']
    
subj_list = ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]
n_subj = len(subj_list) 

fname_suffix = "filter_1_110Hz_notch"
fig_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/preproc_scripts/preproc_ica_EEG_figs/"


start, stop = 000.0, 100.0
# TO BE MODIFIED: mne-python 0.10
# Also check the ics for muscle contaminations

if True:
    i = 0
    subj = subj_list[i]+"_EEG"
    print subj  
    fif_name = filter_dir  + "%s/%s_%s_raw.fif" %(subj,subj,fname_suffix)
    raw = mne.io.Raw(fif_name,preload = True)
    raw = raw.crop(tmin = start, tmax = stop)
    
    # ============= load ICA =========================================================
    ica_name = ica_dir  + "%s/%s_%s_raw_ica_obj-ica.fif" %(subj,subj,fname_suffix)
    ica = mne.preprocessing.read_ica(ica_name)
    mat_name = ica_dir  + "%s/%s_%s_raw_ECG_EOG.mat" %(subj,subj,fname_suffix)
    EXG_dict = scipy.io.loadmat(mat_name)
    eog_scores = EXG_dict['eog_scores'][0]
    ecg_scores = EXG_dict['ecg_scores'][0]
    
    if len(EXG_dict['eog_inds'])>0:
        eog_inds = EXG_dict['eog_inds'][0]
    else: 
        # if no automatic EOG was detected, use the first 10 ICs that are mostly
        # correlated with EOG1
        print "empty EOG inds"
        eog_inds = np.argsort(np.abs(eog_scores))[-1:-10:-1]
    
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
    
    # Subj5 eog_inds = [53]
    # Subj6 ecg_inds = []
    # Subj8 ecg_inds = [102, 86, 49]
    # Subj10 eog_inds = [2] ecg_inds = []
    # Subj11 eog_inds = [84, 1]
    # Subj12 eog_inds = [45, 87, 85]  ecg_inds = []
    # Subj13 ecg_inds = [19, 54]
    
    
    # there are too many components to visualize, I only select from the automatically detected ones
    ica.plot_scores(eog_scores, exclude=eog_inds)
    tmp_fig_name = fig_dir  + "%s_%s_EOG_score.png" %(subj,fname_suffix)
    plt.savefig(tmp_fig_name)
    plt.close()
    
    ica.plot_scores(ecg_scores, exclude=ecg_inds)
    tmp_fig_name = fig_dir  + "%s_%s_ECG_score.png" %(subj,fname_suffix)
    plt.savefig(tmp_fig_name)
    plt.close()
    
    #Subj14 
    #eog_inds = [83]
    
    t0 = time.time()
    fig1 = ica.plot_sources(raw,picks = eog_inds )
    tmp_fig_name = fig_dir  + "%s_%s_EOG_ts.png" %(subj,fname_suffix)
    fig1.savefig(tmp_fig_name)
    print time.time()-t0
    

    t0 = time.time()
    fig2 = ica.plot_sources(raw, picks = ecg_inds )
    tmp_fig_name = fig_dir  + "%s_%s_ECG_ts.png" %(subj,fname_suffix)
    fig2.savefig(tmp_fig_name)
    print time.time()-t0
    
    # changes after upgrading to MNE 0.10
    # press "-" to reduce scales
    # then re-run the following        
    tmp_fig_name = fig_dir  + "%s_%s_EOG_ts.png" %(subj,fname_suffix)
    fig1.savefig(tmp_fig_name)
    tmp_fig_name = fig_dir  + "%s_%s_ECG_ts.png" %(subj,fname_suffix)
    fig2.savefig(tmp_fig_name)
    
    
    if False:
        # chedk the frequency spectrum
        IC_ts = ica.get_sources(raw)
        tmp_psd, freq = mne.time_frequency.compute_raw_psd(IC_ts)
        [IC_ts_data, times ]= IC_ts[:,:]
        plt.plot(freq, tmp_psd.T)
        plt.figure(); plt.plot(freq, tmp_psd[[7,13, 26, 30, 61, 71, 84]].T)
        # visualize the spectrum of 
        t0 = time.time()
        fig3 = ica.plot_sources(raw)
        #tmp_fig_name = fig_dir  + "%s_%s_muscle_ts.png" %(subj,fname_suffix)
        #fig3.savefig(tmp_fig_name)
        print time.time()-t0
    

    # ================ make manual selections =====================================
    #new_eog_inds = eog_inds[1:2]
    if True:
        new_eog_inds = eog_inds[0:1]
        #new_eog_inds = eog_inds[0:2]
        #new_ecg_inds = ecg_inds[0:3]
        new_ecg_inds = ecg_inds[0:0]
        
        muscle_inds = []
        
        # it is too dangerous to remove the muscle components here.  
        # They are reletively in a short range of frequency, 15 Hz ish
        # But many components have it, it is too risky to remove them
        print new_eog_inds, new_ecg_inds, muscle_inds
        new_mat = dict(new_eog_inds = new_eog_inds, new_ecg_inds = new_ecg_inds, muscle_inds = muscle_inds)
        new_mat_name = ica_dir  + "%s/%s_%s_ECG_EOG_manual_check.mat" %(subj,subj,fname_suffix)
        print new_mat_name
        scipy.io.savemat(new_mat_name, new_mat)
        
    if False:
        del(ica)
        del(raw)
       
if False:
    # print the eog and ecg inds at the end
    for i in range(n_subj):
        subj = subj_list[i]+ "_EEG"
        new_mat_name = ica_dir  + "%s/%s_%s_ECG_EOG_manual_check.mat" %(subj,subj,fname_suffix)
        tmp_dict = scipy.io.loadmat(new_mat_name)
        print tmp_dict['new_eog_inds'], tmp_dict['new_ecg_inds'], tmp_dict['muscle_inds']
        del(tmp_dict)
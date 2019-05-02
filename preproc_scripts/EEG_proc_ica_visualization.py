# -*- coding: utf-8 -*-
"""
Manual scripts, to save the ica components to exclude. 

@author: ying
"""

import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
import scipy.io
import scipy.stats



#====================================================================================
tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
filter_dir =  tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"

EOG_list = ['EOG_LO1','EOG_LO2','EOG_IO1','EOG_SO1','EOG_IO2']
ECG_list = ['ECG']


# ====================for future, these should be written in a text file============
subj_list = [1,2,3,4, 5,6,7,8 10]
n_subj = len(subj_list)

fname_suffix = "filter_1_50Hz"


for i in range(n_subj):
    subj = "Subj%d_EEG" %subj_list[i]
    fig_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/preproc_scripts/preproc_ica_EEG_figs/%s/" % subj 
    print subj   
    # ============= load ICA =========================================================
    ica_name = ica_dir  + "%s/%s_%s_ica_obj-ica.fif" %(subj,subj,fname_suffix)
    ica = mne.preprocessing.read_ica(ica_name)
    out = ica.plot_components()
    for k in range(len(out)):   
        tmp_fig_name = fig_dir  + "%s_%s_ica_topo_set%d.png" %(subj,fname_suffix,k)
        out[k].savefig(tmp_fig_name)
    
    plt.close('all')
    del(ica)
   
    
# check the correlation with neiboring electrodes?
# Figure how to read the positions of the electrodes? or encode an adjacency matrix manually!!
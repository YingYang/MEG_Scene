# -*- coding: utf-8 -*-
import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import time
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy


#=======================================================================================
meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
MEGorEEG = ['EEG','MEG']
# try unsmoothed
#MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"
#EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0"
#EEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"

# MEEG match
MEG_fname_suffix = "1_110Hz_notch_ica_MEEG_match_ave_alpha15.0"
#EEG_fname_suffix = "1_110Hz_notch_ica_MEEG_match_ave_alpha15.0"
EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_MEEG_match_ave_alpha15.0"
print EEG_fname_suffix
#Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14]
#Subj_list = [16, 18]


suffix = "no_aspect"
mat_name = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/aspect_energy.mat"
mat_dict = scipy.io.loadmat(mat_name)
proj =  mat_dict['proj'] 


"""
for isMEG in [0, 1]:
    if isMEG:
        isMEG = 1
        percent = 100
        #Subj_list = range(1,10)
        #Subj_list = range(1,14)
        #Subj_list = range(1,16)
        Subj_list = range(16,19)
        n_times = 110
        n_channels = 306
    else:
        isMEG = 0
        #Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13]
        #Subj_list = [14]
        Subj_list = [16, 18]
        n_times = 109
        n_channels = 128
"""
"""
if True:
    isMEG = 0
    Subj_list = ['SubjYY_100', 'SubjYY_200', 'SubjYY_500']
    Subj_list = ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]
    EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0"
"""


if True:
    # EEG
    #isMEG = False
    #MEG stitch
    subj_ind = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
    
    # MEG
    isMEG = True
    #subj_ind = range(1,19)
    Subj_list = list()
    for i in subj_ind:
        Subj_list.append("Subj%d" %i)    
    
    
    
    
    fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
    print fname_suffix
      
    n_Subj = len(Subj_list)
    n_im = 362
     
    for i in range(n_Subj):            
        #subj = "Subj%d" %Subj_list[i]
        subj = Subj_list[i]
        if isMEG:
            ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
            matname = meg_dir + "%s/%s_%s_%s.mat" %(subj,subj,fname_suffix, suffix)
        else:
            ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
            matname = eeg_dir + "%s_EEG/%s_EEG_%s_%s.mat" %(subj,subj,fname_suffix, suffix)
        mat_dict = scipy.io.loadmat(ave_mat_path)
        data = mat_dict['ave_data']
        data2 = data.copy()
        for j in range(data2.shape[1]):
            data2[:,j,:] = proj.dot(data2[:,j,:])
            
        #data1 = np.dot(proj,data.transpose([1,0,2]))
        #print np.linalg.norm(data2-data1)
        mat_dict1 = deepcopy(mat_dict)
        del(mat_dict)
        mat_dict1['ave_data'] = data2.copy()
        scipy.io.savemat(matname, mat_dict1)
        print subj
   
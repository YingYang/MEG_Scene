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

tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
filter_dir =  tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"
fname_suffix = "filter_1_110Hz_notch_raw"
EOG_list = ['EOG_LO1','EOG_LO2','EOG_IO1','EOG_SO1','EOG_IO2']
ECG_list = ['ECG']
n_eeg_channels = 128

#subj_id_seq = [1,2,3,4,5,6,7,8,10,11,12,13,14, 16, 18]    
###subj_list = ['Extra1','Extra2'] 
#subj_list = list()
#for i in subj_id_seq:
#    subj_list.append('Subj%d' % i)

#subj_list = ["Subj14"]
#subj_list = ["Subj16", "Subj18"]

#subj_list = ['SubjYY_100', 'SubjYY_500']

subj_list = ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]
n_subj = len(subj_list) 

l_freq = 1.0
h_freq = 110.0

fname_suffix = "filter_1_110Hz_notch"

flag_switch_PPO10_POO10 = True
if flag_switch_PPO10_POO10:
    fname_suffix1 = "filter_1_110Hz_notch_PPO10POO10_swapped"
else:
    fname_suffix1 = fname_suffix
#========================= get ICA data===========================================
if True:
    for i in range(n_subj):
        subj = subj_list[i] + "_EEG"
        
        fif_name = filter_dir  + "%s/%s_%s_raw.fif" %(subj,subj,fname_suffix)
        raw = mne.io.Raw(fif_name,preload = True)
        # ============= load ICA =========================================================
        ica_name = ica_dir  + "%s/%s_%s_raw_ica_obj-ica.fif" %(subj,subj,fname_suffix)
        ica = mne.preprocessing.read_ica(ica_name)
        new_mat_name = ica_dir  + "%s/%s_%s_ECG_EOG_manual_check.mat" %(subj,subj,fname_suffix)
        new_mat = scipy.io.loadmat(new_mat_name)
        if len(new_mat['new_eog_inds'])>0:
            new_eog_inds = new_mat['new_eog_inds'][0].astype(np.int)
        else:
            new_eog_inds = []
        if len(new_mat['new_ecg_inds'])>0:
            new_ecg_inds = new_mat['new_ecg_inds'][0].astype(np.int)
        else:
            new_ecg_inds = []
        if len(new_mat['muscle_inds'])>0:
            muscle_inds = new_mat['muscle_inds'][0].astype(np.int)
        else:
            muscle_inds = []    
        
        union = np.union1d(muscle_inds, np.union1d(new_eog_inds, new_ecg_inds))
        exclude = union.astype(np.int).tolist()
        print exclude
        ica.exclude = exclude

        tmp_t = time.time()
        raw_after_ica = ica.apply(raw,exclude = ica.exclude)  
        
        if flag_switch_PPO10_POO10:
            print "swapping PPO10h and POO10h"
            # swap the raw data in PPO10 and POO10
            n_channels = len(raw.info['ch_names'])
            ch_ind_PPO10 = [l for l in range(n_channels) if raw.info['ch_names'][l] == 'PPO10h'][0]
            ch_ind_POO10 = [l for l in range(n_channels) if raw.info['ch_names'][l] == 'POO10h'][0]
            
            # debug tmp1 = raw_after_ica._data[:,0:100].copy()
            
            tmp =  raw_after_ica._data[ch_ind_POO10].copy()
            raw_after_ica._data[ch_ind_POO10] = raw_after_ica._data[ch_ind_PPO10].copy()
            raw_after_ica._data[ch_ind_PPO10] = tmp.copy()
            # debug tmp2 = raw_after_ica._data[:,0:100].copy()
            # debug print tmp1[ch_ind_PPO10]-tmp2[ch_ind_POO10],  tmp1[ch_ind_POO10]-tmp2[ch_ind_PPO10], tmp1[ch_ind_POO10]- tmp2[ch_ind_POO10]
            
            
            # debug
            #raw1 = mne.io.Raw(ica_dir  + "%s/%s_%s_ica_raw.fif" %(subj,subj,fname_suffix1), preload = True)
            #raw0 = mne.io.Raw(ica_dir  + "%s/%s_%s_ica_raw.fif" %(subj,subj,fname_suffix), preload = True)
            #print np.linalg.norm(raw1._data[ch_ind_POO10] - raw0._data[ch_ind_PPO10])
            #print np.linalg.norm(raw1._data[ch_ind_POO10] - raw0._data[ch_ind_POO10])
            #print np.linalg.norm(raw1._data[0] - raw0._data[0])
                
                    
        # interpolate the bad channels here, so it is prepared for epoch computing
        # for SubjYY_100 and SubjYY_500, I did not have digitized locations, and
        # interpolation can not be done
        if subj not in ['SubjYY_100_EEG','SubjYY_500_EEG']:
            raw_after_ica.interpolate_bads(reset_bads = True, mode = "accurate")
            
        ica_raw_name = ica_dir  + "%s/%s_%s_ica_raw.fif" %(subj,subj,fname_suffix1)
        raw_after_ica.save(ica_raw_name, overwrite = True)  
        print time.time()-tmp_t
        
        # =============== clean the objects to release memory ======================
        del(raw)
        del(ica)
        del(raw_after_ica)

    
#================================================================================
# reference
# I do not need rereferencing, by default, the referencing will altomatically be applied to the data
#if False:
#    print "=============== re-reference======================================="
#    Masteroids = ['M1','M2']
#    EOG_list = ['EOG_LO1','EOG_LO2','EOG_IO1','EOG_SO1','EOG_IO2']
#    ECG_list = ['ECG']
#    drop_names = []
#    for i in range(7):
#        drop_names.append("misc%d"%(i+1))
#        trigger_list = ['STI101']      
#    # the trigger is now status 
#    exclude_list = Masteroids + EOG_list + ECG_list + drop_names + trigger_list
#         
#    for i in range(n_subj):
#        subj = "Subj%d_EEG" %subj_list[i]
#        ica_raw_name = ica_dir  + "%s/%s_%s_ica_raw.fif" %(subj,subj,fname_suffix)
#        ica_raw_name_save = ica_dir  + "%s/%s_%s_ica_raw_reref.fif" %(subj,subj,fname_suffix)
#        raw = mne.io.Raw(ica_raw_name, preload = True)  
#        
#        #=============================================
#        picks = mne.pick_channels(raw.info['ch_names'],include = [], exclude = exclude_list + raw.info['bads'])
#        print len(picks)
#        data = raw._data[picks,:]
#        data = (data- np.mean(data, axis = 0))
#        raw._data[picks,:] = data
#        # This can be wrong referencing
#        #=============================================
#        # the mne default re-referencing.  I have not verified it the same as rereferencing for each time point, 
#        # or across time points. it seems to be across all time points, not at each time point
#        raw1, ref_data = mne.io.set_eeg_reference(raw,ref_channels = None, copy = True)
#        raw.save(ica_raw_name_save, overwrite = False)
#        # =============== clean the objects to release memory ======================
#        del(raw)
#        

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



import sys
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
#path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
path1 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
#path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from ols_regression import ols_regression
from Stat_Utility import bootstrap_mean_array_across_subjects
from Script_sensor_results_visualization import visualize_sensors_MEG_topo
# answer Reviwer 2's question


data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
data_root_dir =data_root_dir0 +"MEG_preprocessed_data/MEG_preprocessed_data/epoch_raw_data/"


#=======================================================================================
fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"
#EEG_fname_suffix = "1_110Hz_notch_ica_MEEG_match_ave_alpha15.0"

dropbox_dir = "/media/yy/LinuxData/yy_dropbox/Dropbox/"

suffix = "no_aspect"
mat_name = dropbox_dir + "/Scene_MEG_EEG/Features/Pixel_features/aspect_energy.mat"
mat_dict = scipy.io.loadmat(mat_name)
proj =  mat_dict['proj'] 



X = mat_dict['X']

Subj_list = range(1,19)
n_times = 110
n_channels = 306
   


if True:
      
    n_Subj = len(Subj_list)
    n_im = 362
    
    Rsq = []
     
    for i in range(n_Subj):            
        #subj = "Subj%d" %Subj_list[i]
        subj ="Subj%d"% Subj_list[i]
        ave_mat_path = data_root_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
        matname = data_root_dir + "%s/%s_%s_%s.mat" %(subj,subj,fname_suffix, suffix)
        mat_dict = scipy.io.loadmat(ave_mat_path)
        data = mat_dict['ave_data']
        data2 = data.copy()
        #for j in range(data2.shape[1]):
        #    data2[:,j,:] = proj.dot(data2[:,j,:])
            
        ave_data = data2
        tmp_result = ols_regression(ave_data, X, stats_model_flag = False) 
        
        Rsq.append(tmp_result['Rsq'])
        

mean_Rsq = np.array(Rsq).mean(axis = 0)

mean_Rsq1 = np.reshape(mean_Rsq, [102,3, mean_Rsq.shape[1]]).mean(axis = 1)
mean_Rsq2 = mean_Rsq.copy()
for i in range(3):
    mean_Rsq2[i::3] = mean_Rsq1


fname = "/media/yy/LinuxData/yy_Scene_MEG_data/MEG_preprocessed_data/MEG_preprocessed_data/epoch_raw_data/Subj1/Subj1_run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"
tmp_epoch = mne.read_epochs(fname)
info1 = tmp_epoch.info
times = tmp_epoch.times-0.04
del(tmp_epoch)

data = scipy.io.loadmat("/media/yy/LinuxData/yy_dropbox/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/" \
                              + "Utility/sensor_layout.mat")

sensor_loc = np.array(list(map(lambda x: x['loc'], info1['chs'])))
pos = data['position'][:,0:2]  
r = 0.001
for l in range(306):
    '''
    #info1['chs'][l]['coil_type'] = 3012
    info1['chs'][l]['unit']= 112
    #info1['chs'][l]['loc'][0:2]= pos[l]
    #info1['chs'][l]['loc'][2::] = 0
    if np.mod(l,3) == 0:
        info1['chs'][l]['loc'][1] += r
    else:
        info1['chs'][l]['loc'][1] -= r/2
        if np.mod(l,3) == 1:
            info1['chs'][l]['loc'][1] -= r/2*np.sqrt(3)
        else:
            info1['chs'][l]['loc'][1] += r/2*np.sqrt(3)
    ''' # gives a funny bias, overlap the gradiometer and megnotometer
    pass
'''
ch_names = []
for i in range(306):
    ch_names.append("%d" %i)


pos1 = pos.copy()
pos1 = (pos1-0.5)*100.0
pos1 = np.hstack([pos1,sensor_loc[:,2:3]*250])

montage= mne.channels.Montage(pos1, ch_names, kind = "none", 
                               selection = range(306))
montage.plot()


info2 = mne.create_info(ch_names, sfreq = 100.0, ch_types = "eeg",
                       montage = montage )

'''


fig_outdir = data_root_dir0 + "Result_Mat/figs/sensor_reg/MEEG_no_aspect/"
fig_name = fig_outdir + \
                    "Subj_pooled_aspect.pdf"


times_array = np.arange(0., .500, 0.05)

visualize_sensors_MEG_topo(info1, mean_Rsq*100.0, 
                           vmin = 0, vmax = 15, figname = fig_name, 
                           figsize = (20,10) , 
                      times = times, time_array = times_array,
                      title = "aspect", ylim = [0,15])
# answer Reviwer 2's question
   
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import mne, time
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io
import os
import scipy.misc
import scipy.spatial
import sklearn.manifold

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)

meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"


MEGorEEG = ["EEG","MEG"] 
if False:
    fname_suffix = "1_50Hz_raw_ica_window_50ms_ave_alpha15.0"
    isMEG = 1
    Subj_list = range(1,14)
    n_times = 110

# for eeg only, currently we only have Subj3
if True:
    isMEG = 0
    Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13]
    fname_suffix = "1_50Hz_raw_ica_window_50ms_ave_alpha15.0"
    n_times = 109

# can not use os.listdir() here, they are not alphabetical,  maybe I have used this somewhere else?
image_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Images/"
image_path_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Images/Image_List.txt"
image_list = np.loadtxt(image_path_fname, delimiter = " ", 
                                 dtype = np.str_)
n_im = len(image_list)

n_subj = len(Subj_list)
#time_seq = np.arange(0.1,0.6,0.1)
# half width of the time windows
#time_half_width = 0.025
time_seq = [0.3]
time_half_width = 0.25
n_time_window = len(time_seq)
RDM = np.zeros([n_subj, n_time_window, n_im, n_im])
metric = "correlation"
for i in range(n_subj):
    subj = "Subj%d" %Subj_list[i]
    if isMEG:
        ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
    else:
        ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
        
        #compute correlation
    ave_mat = scipy.io.loadmat(ave_mat_path)
    ave_data = ave_mat['ave_data']
    ave_data -= np.mean(ave_data,axis = 0)
    times = ave_mat['times'][0]
    tmp_RDM = np.zeros([n_time_window,n_im, n_im])
    for t in range(n_time_window):
        tmin, tmax = time_seq[t]-time_half_width, time_seq[t]+time_half_width
        tmp_time_ind = np.all(np.vstack([times>=tmin, times<=tmax]), axis = 0)
        # do not average the data, but concatenate the data
        tmp_data = ave_data[:,:,tmp_time_ind].reshape([n_im, -1])
        tmp_RDM[t] = scipy.spatial.distance.squareform( 
                  scipy.spatial.distance.pdist(tmp_data,metric=metric))
    RDM[i] = tmp_RDM 

mean_RDM = np.mean(RDM, axis = 0) 
mds = sklearn.manifold.MDS(n_components = 2, metric = True, dissimilarity = "precomputed")
pos = np.zeros([n_time_window, n_im, 2])
for t in range(n_time_window):
    pos[t] = mds.fit(mean_RDM[t]).embedding_
    pos[t] = (pos[t] -pos[t].min())/(pos[t].max()-pos[t].min())

shrink = 0.9
pos_to_show = pos*shrink+ (1.0-shrink)/2.0
width = 0.06
if True:
    for t in range(n_time_window):
        fig = plt.figure(figsize = (20,20))
        for i in range(n_im):
            tmp_ax = plt.axes(np.hstack([pos_to_show[t][i,:]-width/2, np.ones(2)*width/2]))
            tmp_im = scipy.misc.imread(image_dir+image_list[i])[50:550,50:550,:]
            tmp_im = scipy.misc.imresize(tmp_im, 0.5)
            _ = plt.imshow(tmp_im)
            _ = plt.axis('off')
        plt.savefig("/home/ying/Dropbox/Scene_MEG_EEG/figs/tmp_MEG%d.pdf" %isMEG)
    
# load the SUN semantic feature, indoor, outdoor, outdoor manmade, R, G, B        
mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/StimSUN_semantic_feat/sun_hierarchy.mat")
X0 = mat_data['data'][:,0:3]  
plt.figure()  
col = ['r','c','b']      
for t in range(n_time_window):
    plt.subplot(1,n_time_window, t+1)
    for i in range(3):
        _=plt.plot(pos_to_show[t][X0[:,i]>0,0],pos_to_show[t][X0[:,i]>0,1], '.'+col[i])



n_rank = 30
mask = np.ones([n_im,n_im])
mask = np.triu(mask,1)
linear_ind = np.arange(n_im*n_im)[np.ravel(mask)>0]
for t in range(n_time_window):
    linear_ind_ind = np.argsort(mean_RDM[t][mask>0])[np.hstack([range(0,n_rank), range(len(linear_ind)-n_rank,len(linear_ind))])]
    two_d_ind = np.unravel_index(linear_ind[linear_ind_ind], [n_im,n_im])
    print two_d_ind
    plt.figure(figsize = (4,80))
    n_pair = len(linear_ind_ind)
    for j in range(n_pair):
        for l in range(2):
            _ = plt.subplot(n_pair,2,2*j+l+1)
            tmp_im = scipy.misc.imread(image_dir+image_list[two_d_ind[l][j]])[50:550,50:550,:]
            tmp_im = scipy.misc.imresize(tmp_im, 0.5)
            _ = plt.imshow(tmp_im)
            _ = plt.axis('off')
        _ = plt.title( "d=%1.3f" % mean_RDM[t][two_d_ind[0][j],two_d_ind[1][j]] )
    #_= plt.title('%1.2f to %1.2f s' %(time_seq[t]-time_half_width, time_seq[t]+time_half_width))
    #plt.tight_layout()
    plt.savefig("/home/ying/Dropbox/Scene_MEG_EEG/figs/tmp_rank_MEG%d.pdf" %isMEG)
                 
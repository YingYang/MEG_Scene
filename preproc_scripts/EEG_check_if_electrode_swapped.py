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



#====================================================================================
tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
filter_dir =  tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/"

EOG_list = ['EOG_LO1','EOG_LO2','EOG_IO1','EOG_SO1','EOG_IO2']
ECG_list = ['ECG']


#=============== the standard theoretical topological map ========================
pos3 = scipy.io.loadmat("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/biosemi128_layout/biosemi128_pos.mat")
pos3 = pos3['pos']
# using the layout matrix to create a distance matrix
n = 128
dist_mat = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        dist_mat[i,j] = np.sqrt(np.sum((pos3[i,:] - pos3[j,:])**2) )

mask = np.ones([n,n])
mask = np.triu(mask, 1)


# threshold the distance matrix with 2,3,4... nearest neighbors
n_neighbor_seq = np.arange(2,3)
m = len(n_neighbor_seq)
thres_mat = np.zeros([m,n,n])
for k in range(m):
    for i0 in range(n):
        tmp_argsort = np.argsort(dist_mat[i0])
        tmp_binary = np.zeros(n)
        tmp_binary[tmp_argsort[0:n_neighbor_seq[k]]] = 1.0
        thres_mat[k,i0] = tmp_binary
        
#======================check digitization =====================================
subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
n_subj = len(subj_list)
fname_suffix = "raw"

# kendall's tau with the standard topology map
ktau = np.zeros(n_subj)
p_ktau = np.zeros(n_subj)
corr = np.zeros(n_subj)
diff_neighbor_mat = np.zeros(n_subj)


for i in range(n_subj):
    subj = "Subj%d_EEG" %subj_list[i]
    
    fif_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/raw_data/%s/%s_%s.fif" %(subj,subj, fname_suffix)
    raw = mne.io.Raw(fif_name)
    # find a way to reade electrode locations
    # the digitization information
    dig = raw.info['dig']  # a list with 131 (128+3) elements, first three, fiducials
    coord = np.zeros([len(dig),3])
    for k in range(len(dig)):
        coord[k] = dig[k]['r']
      
    tmp_dist_mat = np.zeros([n,n])
    for i0 in range(n):
        for j0 in range(n):
            # skip the first 3 fidicual points
            tmp_dist_mat[i0,j0] = np.sqrt(np.sum((coord[3+i0,:] - coord[3+j0,:])**2) )   
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(dist_mat, interpolation = "none")
    plt.title('standard topo %s' %subj)
    plt.subplot(1,2,2)
    plt.imshow(tmp_dist_mat, interpolation = "none")
    plt.title('digitization %s' %subj)
    
    ktau[i], p_ktau[i] = scipy.stats.kendalltau(dist_mat[mask>0], tmp_dist_mat[mask>0])
    tmpcorr = np.corrcoef(dist_mat[mask>0], tmp_dist_mat[mask>0])
    corr[i] = tmpcorr[0,1]
    
    tmp_thres_mat = np.zeros([m,n,n])
    for k in range(m):
        for i0 in range(n):
            tmp_argsort = np.argsort(tmp_dist_mat[i0])
            tmp_binary = np.zeros(n)
            tmp_binary[tmp_argsort[0:n_neighbor_seq[k]]] = 1.0
            tmp_thres_mat[k,i0] = tmp_binary
    diff_neighbor_mat[i] = np.sum(np.abs(tmp_thres_mat-thres_mat))    
    del(raw)    
                
    if False: # manual check
        from mpl_toolkits.mplot3d import Axes3D
        plt.figure()
        ax = plt.subplot(1,1,1, projection='3d')
        ax.plot(coord[0:131,0], coord[0:131,1], coord[0:131,2], '.')
        for k in range(128):
            _ = ax.text(coord[3+k,0], coord[3+k,1], coord[3+k,2], raw.info['ch_names'][k])
        
        #  2D plot, comment
        #plt.figure()
        #plt.plot(coord[3::,0], coord[3::,1], '.')
        #for k in range(128):
        #    plt.text(coord[3+k,0], coord[3+k,1], raw.info['ch_names'][k])
        
    #========== notes =======================================================
    # Mannual checking order: the two rings near the horizontal equatior
    #  then the medial array
    #  then both left and right side, follow the sequence from medial to lateral
    #  e.g. Cz C1 C3, C5, T7 ....
    # Manually checked subjects:
    # Subj 1,2,3,4,5,6,7,8,10,11,12,13,14,16,18
        
    # Subj14, C4 was flat. # check if this is true, index for C4 is 96.
    # This is correct! To see it, set raw.info['projs'] = [], then do raw.plot
    #  data, times = raw[:,:]
    #  plt.plot(data[0:5, 0:5000].T)

#======== check if the electrodes were plugged in to the wrong slots on the cap ==============



#======================= test Kevin's subject 007 ================================

if False:
    swap = False
    fif_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/Pilots/Obj_valence_electrode_swap/007_Loc.fif"
    raw = mne.io.Raw(fif_name)
    n = 128
    data0, times = raw[0:n,:]
    data1 = data0[:,0:10000]
    
    
    if swap:
        # swap PPO10h POO10h
        #id1 = np.nonzero(np.array(raw.info['ch_names']) == "PPO10h")[0][0]
        #id2 = np.nonzero(np.array(raw.info['ch_names']) == "POO10h")[0][0]
        id1 = np.nonzero(np.array(raw.info['ch_names']) == "F1")[0][0]
        id2 = np.nonzero(np.array(raw.info['ch_names']) == "Iz")[0][0]
        tmp = data1[id1,:]
        data1[id1,:] = data1[id2,:]
        data1[id2,:] = tmp 
    
    
    corr_coef = np.corrcoef(data0)
    ch_names = raw.info['ch_names'][0:n]
    
    events = mne.find_events(raw, stim_channel = "STI101")
    tmp_epoch = mne.Epochs(raw, events, event_id = [9961473], tmin = -0.1, tmax = 0.1, baseline = (None,0), picks = range(0,n))
    evoked = tmp_epoch.average()
    
    
    layoutpath = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/biosemi128_layout/"
    layout = mne.channels.read_layout("biosemi128", layoutpath) 
    evoked.plot_topomap(layout = layout)
    
    pos = layout.pos[:,0:2]
    # toplogy plots of electrodes
    plt.figure()
    for i in range(n):
        _ = plt.plot(pos[i,0], pos[i,1], '*g')
        _ = plt.text(pos[i,0], pos[i,1],  layout.names[i])
        
    
    pos3 = scipy.io.loadmat("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/biosemi128_layout/biosemi128_pos.mat")
    pos3 = pos3['pos']
    # using the layout matrix to create a distance matrix
    n = 128
    dist_mat = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            dist_mat[i,j] = np.sqrt(np.sum((pos3[i,:] - pos3[j,:])**2) )
            
    # correlation with the nearest k neighbors
    k = 1
    corr_knn = np.zeros(n)
    for i in range(n):
        knn_ind = np.argsort(dist_mat[i,:])[1:k+1]
        #print i in knn_ind  # check the channel itself is not in it 
        tmp_corrcoef = corr_coef[i,:]
        corr_knn[i] = np.mean(tmp_corrcoef[knn_ind])
        
    plt.figure()
    plt.hist(corr_knn, 130)
    
    print np.argsort(corr_knn)[0:2]
    # fake evoked data
    evoked.data = np.tile(corr_knn, [evoked.data.shape[1],1]).T
    evoked.plot_topomap(layout = layout)
    
    refid = np.nonzero(np.array(ch_names) == "Cz")[0][0]
    evoked.data = np.tile(corr_coef[refid,:], [evoked.data.shape[1],1]).T
    evoked.plot_topomap(layout = layout)
    
    
    
                   

import numpy as np
import matplotlib
#matplotlib.use('Agg')

import time

import matplotlib.pyplot as plt
import scipy.io

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)

path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)

import sklearn 
import sklearn.neighbors



meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

MEG_fname_suffix = '1_110Hz_notch_ica_MEEG_match_ave_alpha15.0_no_aspect';
EEG_fname_suffix = '1_110Hz_notch_ica_PPO10POO10_swapped_MEEG_match_ave_alpha15.0_no_aspect';
MEGorEEG = ["EEG","MEG"] 


#Subj_list = range(1,19)
#n_Subj = len(Subj_list)
#Subj_has_EEG = np.ones([n_Subj], dtype = np.bool)
#Subj_has_EEG[[8,14,16]] = False

Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
n_Subj = len(Subj_list)
Subj_has_EEG  = np.ones([n_Subj], dtype = np.bool)

common_times = np.round(np.arange(-0.1, 0.8, 0.01), decimals = 2) 
n_times = len(common_times)
#MEG_offset = 0.04
MEG_offset = 0.02

fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_decoding/"  

# ============define my own distance fuction===============
def correlation_dist(x1,x2):
    return 1- np.corrcoef(x1,x2)[0,1]
#=======================================================

K = 3
#knnclf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, 
#                weights='uniform', algorithm='auto', metric='pyfunc',
#                func = correlation_dist)
knnclf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=3, 
                weights='uniform', algorithm='auto', metric='euclidean')




n_im = 362
n_perm = 40
perm_seq = np.zeros([n_perm+1, n_im], dtype = np.int)
orig_seq = range(0, n_im)
perm_seq[0] = orig_seq
for i in range(n_perm):
    perm_seq[i+1] = np.random.permutation(orig_seq)
n_perm1 = n_perm+1


acc = np.zeros([2, n_Subj, n_perm1, n_times])
n_im = 362
for isMEG in [0,1]:
    fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
    n_channel = 306 if isMEG else 128
    tmp_data = np.zeros([n_Subj, n_im, n_channel, n_times] )
    tmp_Rsq = np.zeros([n_Subj, n_channel, n_times])
    for j in range(n_Subj):
        subj = "Subj%d" %Subj_list[j]
        print subj, MEGorEEG[isMEG]
        if isMEG:
            ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
        else:
            if Subj_has_EEG[j]:
                ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
            else:
                continue
        ave_mat = scipy.io.loadmat(ave_mat_path)
        offset = MEG_offset if isMEG else 0.0
        times = np.round(ave_mat['times'][0] - offset, decimals = 2)
        time_ind = np.all(np.vstack([times <= common_times[-1], times >= common_times[0]]), axis = 0)
        ave_data = ave_mat['ave_data'][:, :, time_ind]
        ave_data -= ave_data.mean(axis = 0) 
        tmp_data[j,:,:,:] = ave_data
    
    
    
    for k in range(n_perm1):
        image_seq = (perm_seq[k]).astype(np.int)
        
        for j in range(n_Subj):  
            if isMEG:
                valid_subj = range(n_Subj)  
            elif Subj_has_EEG[j]:
                valid_subj = np.nonzero(Subj_has_EEG)[0]
            else:
                continue 
            tmp1 = tmp_data[j,:,:,:]
            other_subj = np.setdiff1d(valid_subj, j)
            tmp2 = tmp_data[other_subj,:,:,:]
            
            t0 = time.time()
            for t in range(n_times):
                # create training and testing samples
                testX = tmp1[:,:,t]
                trainX = tmp2[:,:,:,t].reshape([-1, tmp1.shape[1]])
                testY = (image_seq).astype(np.int)
                trainY = np.tile(image_seq, len(other_subj))
                
                # normalize the norm to 1 for each image, if eucledian distance
                #testX = ((testX.T)/np.sqrt( np.sum(testX**2, axis = 1))).T
                #trainX = ((trainX.T)/np.sqrt( np.sum(trainX**2, axis = 1))).T
                knnclf.fit(trainX, trainY)
                pred= knnclf.predict(testX)
                acc[isMEG,j, k, t] = (pred == testY).mean()
            
            print time.time()-t0
            print j

mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/"\
         + "sensor_decoding/image_id_%dNN_acc_leave_one_subj_out_MEEG.mat" %K 
mat_dict = dict(acc = acc, common_times = common_times, n_times = n_times, 
            MEG_offset = MEG_offset, Subj_list = Subj_list, 
            Subj_has_EEG = Subj_has_EEG, perm_seq = perm_seq )
scipy.io.savemat(mat_name, mat_dict)




if False:
    
    mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/"\
         + "sensor_decoding/image_id_%dNN_acc_leave_one_subj_out_MEEG.mat" %K 
    mat_dict = scipy.io.loadmat(mat_name)
    
    sys.path.insert(0, "/home/ying/Dropbox/MEG_source_loc/Face_Learning_Data_Ana/") 
    from Stat_Utility import excursion_perm_test_1D
    
    acc = mat_dict['acc']
    times = mat_dict['common_times'][0]*1000.0         
    alpha = 0.05
    percentile_list = [alpha/2.0*100.0, (1-alpha/2.0)*100]
    
    acc = acc.mean(axis = 1)
   
    col_seq = ['g','r']
    for isMEG in [False, True]:
        ax = plt.subplot(1,1,1)
        tmp_acc_intact = acc[isMEG, 0]
        tmp_acc_perm = acc[isMEG, 1::]
        lb,ub = np.percentile(tmp_acc_perm, percentile_list, axis = 0)
        _= ax.plot(times, tmp_acc_intact)
        _ = ax.fill_between(times, ub, lb, facecolor='b', alpha=0.4)  
        thresh = np.median(ub)
        clusters, integral, p_val_clusters = excursion_perm_test_1D(tmp_acc_intact, tmp_acc_perm, thresh, tail = 0)
        
    
    plt.xlabel('time (ms)')
    plt.ylabel('decoding accuracy')
    plt.legend(MEGorEEG)
    plt.savefig(fig_outdir+ "sensor_image_id_%dNN_decoding.pdf" % K)



    

    
    
    
    
 

    
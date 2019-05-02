import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import scipy.io
path = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path)
from PLS import  get_PLS_cv_error

meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

subj_list = np.hstack([range(1,9),range(10,14)])
n_subj = len(subj_list)
relative_MSE_all = np.zeros(n_subj, dtype = np.object)
MSE_all = np.zeros(n_subj, dtype = np.object)
#offset_seq = np.arange(0.01, 0.06, 0.01)
offset_seq = np.array([0.02])
n_offset = len(offset_seq)
n_modality = 2
n_im = 362
n_fold = 6
cv_ind = np.random.permutation(np.tile(range(0,n_fold),n_im//n_fold+1))
cv_ind = (cv_ind[0:n_im]).astype(np.int)
#n_component_seq = np.arange(10,40,10)
n_component_seq = np.array([3])

#debug
#n_subj = 3
#n_offset = 1


err = np.zeros([n_subj, n_offset], dtype = np.object)
relative_error = np.zeros([n_subj, n_offset], dtype = np.object)
n_comp_best_all = np.zeros([n_subj, n_offset], dtype = np.object)
common_times_all = np.zeros(n_offset, dtype = np.object)

for l in range(n_offset):
    offset = offset_seq[l]
    # one Subject
    for i in range(n_subj):    
        subj = "Subj%d" %subj_list[i]
        MEG_fname_suffix = "1_50Hz_raw_ica_window_50ms_ave_alpha15.0"
        MEG_ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,MEG_fname_suffix)
        data = scipy.io.loadmat(MEG_ave_mat_path)
        MEG_time_ms = ((data['times'][0]-offset)*1000).astype(np.int)
        MEG_data = data['ave_data']
        MEG_picks_all = data['picks_all'][0]
        del(data)
        
        EEG_fname_suffix = "1_50Hz_raw_ica_window_50ms_ave_alpha15.0"
        EEG_ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,EEG_fname_suffix)
        data = scipy.io.loadmat(EEG_ave_mat_path)
        EEG_time_ms = (data['times'][0]*1000).astype(np.int)
        EEG_data = data['ave_data']
        EEG_picks_all = data['picks_all'][0]
        del(data)
        
        #==================================================================================
        # take the intersection of time points
        common_times  = np.intersect1d(MEG_time_ms, EEG_time_ms)
        MEG_in_common_time_id = [l0 for l0 in range(len(MEG_time_ms)) if MEG_time_ms[l0] in common_times]
        EEG_in_common_time_id = [l0 for l0 in range(len(EEG_time_ms)) if EEG_time_ms[l0] in common_times]
        MEG_data = MEG_data[:,:,MEG_in_common_time_id]
        EEG_data = EEG_data[:,:,EEG_in_common_time_id]
        
        # scale the MEEG data 
        # Should I normalized different sensors? If so, the MEG prediction looks terrible
        # Try just divided the data by the 10E-12 and 10E-13
        # Try just use magnetometers, it was fine.
        EEG_data -= np.mean(EEG_data, axis = 0)
        EEG_data /= np.std(EEG_data.ravel())
        # normalize the two gradiometers and magnotometers seperately
        MEG_data -= np.mean(MEG_data, axis = 0)
        for ll in range(3):
            MEG_data[:,ll::3,:] /= np.std( np.ravel(MEG_data[:,ll::3,:])) *np.sqrt(3)
        
        #MEG_data = MEG_data[:,1::3,:]
        n_times = len(common_times)
        MEEG_data = [MEG_data, EEG_data]
        tmp_err = np.zeros([n_modality,n_times])
        tmp_relative_error = np.zeros([n_modality,n_times])
        tmp_n_comp_best = np.zeros([n_modality,n_times])
        tmp_score = np.zeros([n_modality,n_times])
     
        for k in range(n_modality):
            indX = k
            indY = np.setdiff1d(np.arange(n_modality).astype(np.int), np.array([indX]))[0]
            print subj, indX, indY
            for t in range(n_times):
                # use the current modality as the current modality
                # debug, use only the magnetometers
                X = MEEG_data[indX][:,:,t]
                Y = MEEG_data[indY][:,:,t]
                tmp_result = get_PLS_cv_error(X, Y, cv_ind, n_component_seq, B = 3, scale = True)
                tmp_err[k,t] = tmp_result['errY']
                tmp_relative_error[k,t] = tmp_result['relative_errY']
                tmp_n_comp_best[k,t] = tmp_result['n_comp_best'] 
                tmp_score[k,t] = tmp_result['score']
        
        common_times_all[l] = common_times
        err[i,l] = tmp_err
        relative_error[i,l] = tmp_score
        n_comp_best_all[i,l] = tmp_n_comp_best
        
 
err_mat = np.zeros(n_offset, dtype = np.object)
relative_err_mat =  np.zeros(n_offset, dtype = np.object)
n_comp_best_mat = np.zeros(n_offset, dtype = np.object)

for l in range(n_offset):
    tmp_err_mat = np.zeros([n_subj, n_modality, len(common_times_all[l])])
    tmp_relative_err_mat = np.zeros([n_subj, n_modality, len(common_times_all[l])])
    tmp_n_comp_best_mat = np.zeros([n_subj, n_modality, len(common_times_all[l])])
    for i in range(n_subj):
        tmp_err_mat[i] = err[i,l]
        tmp_relative_err_mat[i] = relative_error[i,l]
        tmp_n_comp_best_mat[i] = n_comp_best_all[i,l]
    err_mat[l] = tmp_err_mat
    relative_err_mat[l] = tmp_relative_err_mat
    n_comp_best_mat[l] = tmp_n_comp_best_mat
    
    
mat_dict = dict(err_mat = err_mat, relative_err_mat = relative_err_mat, 
                n_comp_best_mat = n_comp_best_mat,
                common_times_all =common_times_all,
                subj_list = subj_list, n_component_seq = n_component_seq)
                
mat_name = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/MEEG_comp/"
scipy.io.savemat(mat_name + "PLS.mat", mat_dict)  

#================================================================================

mat_dict = scipy.io.loadmat(mat_name + "PLS.mat")
err_mat = mat_dict['err_mat']
relative_err_mat = mat_dict['relative_err_mat']
n_comp_best_mat = mat_dict['n_comp_best_mat'] 
common_times_all =  mat_dict['common_times_all']
    
plt.figure()
vmin = 0.6
vmax = 3
n_offset = 1
for l in range(n_offset):
    for j in range(n_modality):
        plt.subplot(n_offset,2,2*l+j+1)
        plt.imshow(relative_err_mat[0,l][:,j], aspect = "auto",
                   interpolation = "none", 
                   extent = [common_times_all[0,l][0,0], common_times_all[0,l][0,-1], 0,n_subj],
                   origin = "lower",
                   vmin = vmin, vmax = vmax)
        plt.colorbar() 
        plt.title("offset %1.2f" % offset_seq[l])          

plt.figure()
for j in range(n_modality):
    plt.subplot(1,2,j+1)
    for l in range(n_offset):
        plt.plot(common_times_all[0,l][0], np.mean(relative_err_mat[0,l][:,j],axis = 0), '*-', lw = 2)
        plt.legend(offset_seq)
        plt.grid()          

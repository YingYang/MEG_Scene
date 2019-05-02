# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:14:03 2015

@author: ying
"""

import numpy as np
import scipy.io
import scipy.stats
import time

def get_design_matrix(subj, datadir, fname_suffix, feat_name, outdir = None):
    
    """
    Parameters:
    # mat_file_name: the file path with no ".mat" extension
    # e.g
    datadir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
    subj = "Subj1" 
    fname_suffix = "1_50Hz_raw_ica_window_50ms"
    feat_name = "neil_attr"
    
    Return 0, save the design matrix X into a matrix
    """
    
    if outdir is None:
        outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"

    mat_file_name = datadir + "%s/%s_%s_all_trials" %(subj, subj, fname_suffix)    
    # load mat_file_name
    mat_dict = scipy.io.loadmat(mat_file_name + ".mat")
    im_id_no_repeat = mat_dict['im_id_no_repeat'][0] - 1
    
    # load the regressor
    regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
    regressor_fname = regressor_dir +  "%s_PCA.mat" %(feat_name)
    print regressor_dir
    regressor_dict  = scipy.io.loadmat(regressor_fname)
    X0 = regressor_dict['X']
    print regressor_dict['feat_name']
    X_all_trials = np.zeros([len(im_id_no_repeat), X0.shape[1]])
    
    
    n_im = 362

    for i in range(n_im):
        tmp_ind = np.nonzero(im_id_no_repeat == i)[0]
        if len(tmp_ind) >0:
            X_all_trials[tmp_ind,:] = np.tile(X0[i:i+1,:], [len(tmp_ind), 1])
        else:
            print "im %d not found" %i
            print mat_file_name
    
    # zscore
    X = scipy.stats.zscore(X_all_trials)
    mat_dict = dict(X = X, feat_name = feat_name, var_per_cumsum = regressor_dict['var_per_cumsum'][0])
    mat_name = outdir + "%s_%s_dm_PCA.mat" % (subj,feat_name)
    scipy.io.savemat(mat_name, mat_dict)
     
    return 0
    
   
#================================================================================
if __name__ == '__main__':
    
    import time
    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"

    for feat_name in ["neil_attr", "neil_low", "neil_scene",
                 "conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob", "rawpixel"]:
        datadir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
        fname_suffix = "1_50Hz_raw_ica_window_50ms" 
   
        Subj_list = range(1,10)
        n_subj = len(Subj_list)
        for i in range(n_subj):
            subj = "Subj%d" % Subj_list[i] 
            t0 = time.time()
            get_design_matrix(subj, datadir, fname_suffix, feat_name, outdir = outdir)
            print time.time() - t0
            print subj, feat_name    
    

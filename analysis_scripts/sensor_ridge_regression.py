import numpy as np
import scipy.io
import scipy.stats


import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
from ridge_reg_cv import ridge_reg_cv

import time

def sensor_ridge_regression(subj, datadir, fname_suffix,
			 feat_name, train_im_id, test_im_id,
			 outdir = None,  
			 var_percent = 0.9,
                alpha_seq = None):
    """
    Example input:
        datadir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
        subj = "Subj1" 
        fname_suffix = "1_50Hz_raw_ica_window_50ms"
        feat_name = "neil_attr"
        train_im_id = range(0,180)
        test_im_id = range(180,362)
        var_percent: if 0 to 1, the percent of variance
                    if >1, it is the number of dims to be used!
    """
    if outdir is None:
        outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/"
    mat_file_name = datadir + "%s/%s_%s_all_trials.mat" %(subj, subj, fname_suffix)
    mat_dict = scipy.io.loadmat(mat_file_name)
    im_id_no_repeat = (mat_dict['im_id_no_repeat'][0]-1).astype(np.int)
    picks_all = mat_dict['picks_all'][0]
    times = mat_dict['times'][0]
    epochs_mat_no_repeat = mat_dict['epoch_mat_no_repeat']

    # load the design matrix
    regressor_name = "/home/ying/dropbox_unsync/" \
                   + "MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/"\
                   + "regressor/"\
                   + "%s_%s_dm_PCA.mat" % (subj,feat_name)
    tmp = scipy.io.loadmat(regressor_name)
    X0 = tmp['X']
    if var_percent > 1:
        n_dim = np.int(min(X0.shape[1], var_percent))
        suffix = "%d_dim" % n_dim
    else:
        n_dim = np.nonzero(tmp['var_per_cumsum'][0] >=var_percent)[0][0]
        suffix = "%d_percent" % np.int(var_percent*100)
    
    print suffix, n_dim
    X = X0[:,0:n_dim]
    

    # if I need to remove any of the bad trials, remove them here!!!
    # remove the corresponding ones from X, epoch_mat_no_repeat, im_id_no_repeat

    is_train = np.array( [im_id_no_repeat[i] in train_im_id for i in range(len(im_id_no_repeat))])
    is_test = np.array( [im_id_no_repeat[i] in test_im_id for i in range(len(im_id_no_repeat))])
    
    X_train = X[is_train,:]
    X_test = X[is_test,:]

    n_times = epochs_mat_no_repeat.shape[2]
    Y = np.reshape(epochs_mat_no_repeat, [epochs_mat_no_repeat.shape[0],-1])
    Y_train = Y[is_train,:]
    Y_test = Y[is_test,:]

    if alpha_seq is None:
        alpha_seq = 10.0**(np.arange(-3,4))
    error_ratio, best_alpha = ridge_reg_cv(X_train, X_test, Y_train, Y_test, alpha_seq, nfold = 2)
    error_ratio = np.reshape(error_ratio, [len(picks_all), n_times])
    best_alpha = np.reshape(best_alpha, [len(picks_all), n_times])

    save_mat = dict(picks_all = picks_all, subj = subj, 
                   train_im_id = train_im_id, test_im_id = test_im_id,
                   error_ratio = error_ratio, best_alpha = best_alpha, 
                   alpha_seq = alpha_seq, times = times)
    mat_name = outdir + "%s_%s_%s_%s_all_trials.mat" \
               %(subj, fname_suffix, feat_name, suffix)
    scipy.io.savemat(mat_name, save_mat, oned_as = "row")
    return 0
    

def sensor_ridge_regression_ave(data, X, train_im_id, test_im_id,
                alpha_seq = None):
    """
    data [n_im, n_channel, n_times]
    """
    X_train = X[train_im_id,:]
    X_test = X[test_im_id,:]

    Y = np.reshape( data, [data.shape[0], -1])
    Y_train = Y[train_im_id,:]
    Y_test = Y[test_im_id,:]
    
    # demean the train-test data and X seperately
    X_train -= X_train.mean(axis = 0)
    X_test -= X_test.mean(axis = 0)
    Y_train -= Y_train.mean(axis = 0)
    Y_test -= Y_test.mean(axis = 0)

    if alpha_seq is None:
        alpha_seq = 10.0**(np.arange(-3,4))
    error_ratio, best_alpha = ridge_reg_cv(X_train, X_test, Y_train, Y_test, alpha_seq, nfold = 2)
    error_ratio = np.reshape(error_ratio, [data.shape[1], data.shape[2]])
    best_alpha = np.reshape(best_alpha, [data.shape[1], data.shape[2]])
    return error_ratio, best_alpha   
    

#==========================================================================
if __name__ == '__main__':
    
    """
    20160713: >50% variance explained, for most of the regressors, but it could be due to image identity
    
    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/ridge_reg/MEG/" 
    datadir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"   
    fname_suffix = "1_50Hz_raw_ica_window_50ms"
    # create the train-test split:
    n_im = 362
    n_category = n_im//2
    n_train_category = 100
    train_category_ind = np.random.choice(range(n_category), n_train_category, replace = False)
    train_im_id = np.union1d(2*train_category_ind, 2*train_category_ind+1)
    test_im_id = np.setdiff1d(np.arange(0,n_im), train_im_id)
    alpha_seq = 10.0**(np.arange(2,10,3))
    
    Subj_list = np.arange(1,10)
    n_subj = len(Subj_list)
    feat_name_seq = ["neil_attr", "neil_low", "neil_scene",
                 "conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]
    n_feat_name = len(feat_name_seq)             
    # simple testing
    if False:
        for i in range(n_subj):
        #for i in range(1):
            subj = "Subj%d" % Subj_list[i]
            for j in range(n_feat_name):
                feat_name = feat_name_seq[j]
                print subj, feat_name
                sensor_ridge_regression(subj, datadir, fname_suffix,
    			 feat_name, train_im_id, test_im_id, outdir = outdir,  
    			 var_percent = 16,
                             alpha_seq = alpha_seq)
     
    #============================== load the results
    suffix = "80_percent"
    n_channels = 306
    n_times = 110
    error_ratio = np.zeros([n_feat_name, n_subj, n_channels, n_times])
    for j in range(n_feat_name):
        feat_name = feat_name_seq[j]
        for i in range(n_subj):
            subj = "Subj%d"%Subj_list[i]
            # ==================== load the data      
            mat_name = outdir + "%s_%s_%s_%s_all_trials.mat" \
                       %(subj, fname_suffix, feat_name, suffix)
            mat_dict = scipy.io.loadmat(mat_name)
            error_ratio[j,i,:,:] = mat_dict['error_ratio']
    
    import matplotlib.pyplot as plt
    plt.figure(figsize = (8,8))
    vmin, vmax = 0,0.5
    for j in range(n_feat_name):
        _= plt.subplot(3,4,j+1)
        _= plt.imshow(1.0-error_ratio[j].mean(axis = 0), vmin = vmin, vmax = vmax, 
                   interpolation = "none", aspect = "auto")
        _= plt.title( feat_name_seq[j])
        _= plt.colorbar()
    """
    
    # I probably should do cross validation
    meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
    eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
    MEGorEEG = ['EEG','MEG']
    # try unsmoothed
    fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    n_im = 362
    
    
    image_id = (np.arange(n_im).reshape([n_im//2,2])).astype(np.int)
    
    from sklearn.cross_validation import KFold
    n_folds = 5
    kf = KFold(image_id.shape[0], n_folds = n_folds) 
    
    X_list = list()
    local_contrast_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Features/" \
                        + "Layer1_contrast/Stim_images_Layer1_contrast_noaspect.mat"
    X = scipy.io.loadmat(local_contrast_fname)['contrast_noaspect']
    X -= np.mean(X, axis = 0)
    U,D,V = np.linalg.svd(X)
    X_list.append(U)
    
    X_names = ['local_contrast', 'conv1', 'fc6', 'prob']
    
    regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
    for name in X_names[1::]:
        print name
        mat_dict = scipy.io.loadmat(regressor_dir + "AlexNet_%s_no_aspect_PCA.mat" %name)
        X_list.append(mat_dict['X'])

    
    n_feat = len(X_names)
    alpha_seq = 10.0**(np.arange(-4,5,1))    

    outdir =  "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/ridge_reg/"
    n_dim = 100             
    
    #for isMEG in [0, 1]:
    for isMEG in [1]:
        if isMEG:
            Subj_list = range(1,19)
            n_times = 110
            n_channels = 306
        else:
            Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
            n_times = 109
            n_channels = 128
        for i0 in range(1):
        #for i0 in range(1, n_feat):
            # debug 
            X = X_list[i0][:,0:n_dim]
            #X = X_list[i0]               
            n_Subj = len(Subj_list)
            for i in range(n_Subj):
                subj = "Subj%d" %Subj_list[i]
                if isMEG:
                    ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
                else:
                    ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
                
                data = scipy.io.loadmat(ave_mat_path)['ave_data']
                data -= np.mean(data, axis = 0)
                
                # down sample the data to save some time
                data = data[:,:,::2]
                
                error_ratio_kfolds = np.zeros([n_folds, data.shape[1], data.shape[2]])
                counter = 0
                for train_index, test_index in kf:
                    test_im_id = image_id[test_index,:].ravel()
                    train_im_id = image_id[train_index,:].ravel()
                    t0 = time.time()
                    error_ratio, best_alpha =  \
                               sensor_ridge_regression_ave(data, X, \
                               train_im_id, test_im_id, alpha_seq)
                    error_ratio_kfolds[counter] = error_ratio           
                    counter += 1
                    print time.time()-t0
                    print counter, subj
                
                mat_dict = dict(X_name =X_names[i0], X = X, subj = subj,
                                isMEG = isMEG, error_ratio_kfolds = error_ratio_kfolds)
                mat_name = outdir+ "%s_isMEG%d_%s_ridgereg_ndim%d.mat" %(subj, isMEG, X_names[i0], n_dim)
                scipy.io.savemat(mat_name, mat_dict)
                

    """       
    # load the data
    for isMEG in [0,1]:
        if isMEG:
            Subj_list = range(1,19)
            n_times = 110//2
            n_channels = 306
            times = np.arange(-0.1, 1.0, 0.02)-0.04
        else:
            Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
            n_times = 109//2+1
            n_channels = 128
            times = np.arange(-0.1, 1.0, 0.02)
        n_Subj = len(Subj_list)
        for i0 in range(n_feat):
            tmp_acc = np.zeros([n_Subj, n_channels, n_times])
            for i in range(n_Subj):
                subj = "Subj%d" %Subj_list[i] 
                mat_name = outdir+ "%s_isMEG%d_%s_ridgereg_ndim%d.mat" %(subj, isMEG, X_names[i0], n_dim) 
                mat_dict = scipy.io.loadmat(mat_name)
                tmp_acc[i,:,:] = mat_dict['error_ratio_kfolds'].mean(axis = 0)
      """        
        
		  

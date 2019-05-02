
import numpy as np
import scipy.io 
import matplotlib.pyplot as plt



n_dim_contrast = 100 # 80% variance

regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"

feature_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Features/AlexNet_Features/"

layers = ["conv1","conv2","conv3","conv4","conv5","fc6","fc7","prob"]

for flag_no_aspect in [True, False]:
    
    if flag_no_aspect:
        contrast_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/"\
                   +"Stim_images_Layer1_contrast_noaspect.mat"
        mat_dict_contrast = scipy.io.loadmat(contrast_fname)
        contrast = mat_dict_contrast['contrast_noaspect']
    else:
        contrast_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/"\
                   +"Stim_images_Layer1_contrast.mat"
        mat_dict_contrast = scipy.io.loadmat(contrast_fname)
        contrast = mat_dict_contrast['contrast']
        
    contrast -= np.mean(contrast, axis = 0)
    u,d,v = np.linalg.svd(contrast, full_matrices = False)
    u = u[:,0:n_dim_contrast]
    proj = np.eye(u.shape[0]) - u.dot( np.linalg.inv(u.T.dot(u))).dot(u.T)
    plt.figure()
    plt.plot( np.cumsum(d**2) / np.sum(d**2), '-+'); plt.grid();
    
    for layer in layers:
        if flag_no_aspect:
            feat_fname = feature_dir + "AlexNet_%s_no_aspect.mat" %layer
        else:        
            feat_fname = feature_dir + "AlexNet_%s.mat" %layer
            
        mat_dict_feat = scipy.io.loadmat(feat_fname)
        feature = mat_dict_feat['data']
        feature -= np.mean(feature, axis = 0)
        feature = proj.dot(feature)
        
        u0,d0,v0 = np.linalg.svd(feature, full_matrices = False)
        var_per_cumsum = np.cumsum(d0**2)/np.sum(d0**2) 
        mat_dict = dict(X = u0, var_per_cumsum = var_per_cumsum)
    
        if flag_no_aspect:
            mat_name = regressor_dir + "AlexNet_%s_no_aspect_no_contrast%d_PCA.mat" %(layer, n_dim_contrast)
        else:
            mat_name = regressor_dir + "AlexNet_%s_no_contrast%d_PCA.mat" %(layer, n_dim_contrast)
        
        scipy.io.savemat(mat_name, mat_dict)
            
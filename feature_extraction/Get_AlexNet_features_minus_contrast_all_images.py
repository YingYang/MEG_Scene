
import numpy as np
import scipy.io 
import matplotlib.pyplot as plt

regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
feature_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Features/AlexNet_Features/"
layers = ["conv1","conv2","conv3","conv4","conv5","fc6","fc7","prob"]

# ==========for all images, the local contrast, the aspect part is removed==============
contrast_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/"\
           +"All_images_Layer1_contrast.mat"
contrast_mat_dict = scipy.io.loadmat(contrast_fname)
contrast_noaspect = contrast_mat_dict['contrast_noaspect']
contrast_noaspect -= contrast_noaspect.mean(axis = 0)
u0,d0,v0 = np.linalg.svd(contrast_noaspect, full_matrices = False)
var_explained = np.cumsum(d0**2)/np.sum(d0**2)
plt.figure(); plt.plot(var_explained, '-+')
n_all_im = u0.shape[0]

ndim = 160 # 90% variance of the contrast, verified 
u0 = u0[:,0:ndim]
proj = np.eye(n_all_im)-reduce(np.dot, [u0, np.linalg.inv(u0.T.dot(u0)), u0.T])

model_name = "AlexNet"
# 90% variance of the contrast  
for layer in layers:
    feat_fname = feature_dir + "%s_%s_noaspect_all_images.mat" % (model_name, layer)
        
    mat_dict_feat = scipy.io.loadmat(feat_fname)
    feature = mat_dict_feat['data']
    # all features do not have aspect, so should be already demeaned after removing the aspect
    feature -= np.mean(feature, axis = 0)
    feature = proj.dot(feature)
    
    u0,d0,v0 = np.linalg.svd(feature, full_matrices = False)
    var_per_cumsum = np.cumsum(d0**2)/np.sum(d0**2) 
    mat_dict = dict(X = u0, D = d0, var_per_cumsum = var_per_cumsum, data = feature)
    
    mat_name = regressor_dir + "AlexNet_%s_no_aspect_no_contrast%d_all_im_PCA.mat" %(layer, ndim)
    scipy.io.savemat(mat_name, mat_dict)
    print mat_name
            
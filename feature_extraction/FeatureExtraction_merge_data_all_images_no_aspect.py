# -*- coding: utf-8 -*-

# change directory to the master folder/ caffe_root
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time


n_image = 362
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
       'fc6', 'fc7', 'prob']

#================merge regressors of aspect and energy  
mat_dict = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/aspect_energy.mat")
mat_dict2 = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/aspect_energy_extra_images.mat")
X = np.vstack([mat_dict['X0'], mat_dict2['X0']]);

X1= scipy.stats.zscore(X, axis = 0)  
u,d,v = np.linalg.svd(X1, full_matrices = False)
X_reg = np.ones([X.shape[0],6])
X_reg[:,0:5] = u

# projection matrix  I - X(XTX)^{-1}XT
inv = np.linalg.inv(X_reg.T.dot(X_reg))
proj = np.eye(X.shape[0]) -reduce(np.dot, [X, np.linalg.inv(X.T.dot(X)), X.T])

#=============== concatenate the features and regress the aspect and energy out
feat_path = "/home/ying/Dropbox/Scene_MEG_EEG/Features/" 
for model_name in ["AlexNet"]:
    for layer in layers:     
        mat_name1 = feat_path + "%s_Features/%s_%s.mat" % (model_name, model_name, layer)  
        mat_name2 = feat_path + "%s_Features/%s_Extra_Images_%s.mat" % (model_name, model_name, layer)
        data1 = scipy.io.loadmat(mat_name1)['data']
        data2 = scipy.io.loadmat(mat_name2)['data']
        data = np.vstack([data1, data2])
        data3 = proj.dot(data)
        out_mat_name = feat_path + "%s_Features/%s_%s_noaspect_all_images.mat" % (model_name, model_name, layer)
        scipy.io.savemat(out_mat_name, dict(data = data3))
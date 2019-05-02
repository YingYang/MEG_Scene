# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014
PCA of the regression features
@author: ying
"""

import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
matplotlib.use('Agg')

#import matplotlib.pyplot as plt
import scipy.io
outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"

layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]

#=========== choose features to do PCA ===================
if False:
    feat_name_seq = ["neil_attr", "neil_low", "neil_scene",
                     "rawpixel", "rawpixelgraybox","sun_hierarchy"]
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]
    model_names = ["AlexNet","hybridCNN","placeCNN"]
    for model_name in model_names:
        for layer in layers:
            feat_name_seq.append( "%s_%s"%(model_name, layer))
if True:
    feat_name_seq = []
    model_names = ["AlexNet"]
    suffix = "no_aspect" 
    for model_name in model_names:
        for layer in layers:
            feat_name_seq.append( "%s_%s_%s"%(model_name, layer, suffix))
        

#========================================================
for feat_name in feat_name_seq:
    if feat_name in ["neil_attr", "neil_low", "neil_scene"]:
        mat_data = scipy.io.loadmat('/home/ying/dropbox_unsync/MEG_scene_neil/PTB_Experiment/selected_image_second_round_data.mat');
        neil_attr_score = mat_data['attr_score']
        neil_low_level = mat_data['low_level_feat'] 
        neil_scene_score = mat_data['scene_score']
        if feat_name in ["neil_attr"]:
            X0 = neil_attr_score 
        elif feat_name in ["neil_low"]:
            X0 = neil_low_level 
        elif feat_name in ["neil_scene"]:
            X0 = neil_scene_score
        
    elif feat_name in["rawpixel", "rawpixelgray","rawpixelgraybox"]:
        mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/%s.mat" % feat_name)
        X0 = mat_data['X']
        
    elif feat_name in ['sun_hierarchy']:
        mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/StimSUN_semantic_feat/sun_hierarchy.mat")
        X0 = mat_data['data'] 
        # note the true rank is 16 not 19.
    else:
        model_name = feat_name.split("_")[0] 
        mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/%s_Features/%s.mat" %(model_name, feat_name))
        X0 = mat_data['data']
        
    X = X0 - np.mean(X0, axis = 0)
    # PCA: I tested with a very simple low-rank matrix, u is the same when full_matrices is True or False
    u,d,v = np.linalg.svd(X, full_matrices = False)
    var_per_cumsum = np.cumsum(d**2)/np.sum(d**2) 
    X_proj = u
    mat_dict = dict(X = X_proj, D = d, feat_name = feat_name, var_per_cumsum = var_per_cumsum)
    mat_name = outdir + "%s_PCA.mat" % (feat_name)
    scipy.io.savemat(mat_name, mat_dict)
    print feat_name
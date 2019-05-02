# -*- coding: utf-8 -*-

# change directory to the master folder/ caffe_root
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time


n_image = 1086
layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5',
       'fc6', 'fc7', 'prob']

#=========================================================
for model_name in [ "AlexNet"]: 
    for layer in layers:
        tmp_data = list()
        impath = np.zeros(n_image, dtype = np.str)
        for i in range(n_image):
            outpath = "/home/ying/Dropbox/Scene_MEG_EEG/Features/" \
                    + "%s_Features/%s_Extra_Image_Features/%s_Image%d.mat" %(model_name, model_name, model_name,i)
            result = scipy.io.loadmat(outpath)
            tmp_data.append(result[layer])
            impath[i] = result['image_path'][0]
            del(result)
        
        data_shape = tmp_data[-1].shape
        tmp_data_mat = tmp_data[0].ravel()
        datapath = np.zeros(n_image, dtype = np.str)
        for i in range(1, n_image):
            # each row is one image
            tmp_data_mat = np.vstack([tmp_data_mat,  tmp_data[i].ravel()])
        
        mat_dict = dict(data = tmp_data_mat, data_shape = data_shape)
        mat_outpath ="/home/ying/Dropbox/Scene_MEG_EEG/Features/" \
                    + "%s_Features/%s_Extra_Images_%s.mat" % (model_name, model_name, layer) 
        scipy.io.savemat(mat_outpath, mat_dict)
            
 

#=========== merge data of all images, regress the aspect and energy out======
if False:
    #======================================  
    mat_name = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/aspect_energy.mat"
    mat_dict = scipy.io.loadmat(mat_name)
    proj =  mat_dict['proj'] 
    X = mat_dict['X']
    n_im = 362
    proj1 = np.eye(n_im) -reduce(np.dot, [X, np.linalg.inv(X.T.dot(X)), X.T])
    print np.linalg.norm(proj-proj1)
                      
    suffix = "no_aspect"         
    for model_name in [ "AlexNet"]:
        for layer in layers:
            mat_outpath ="/home/ying/Dropbox/Scene_MEG_EEG/Features/" \
                        + "%s_Features/%s_%s.mat" % (model_name, model_name, layer) 
            mat_dict = scipy.io.loadmat(mat_outpath) 
            print mat_dict['data'].shape
            data1 = np.dot(proj, mat_dict['data']) 
            
            mat_outpath1 ="/home/ying/Dropbox/Scene_MEG_EEG/Features/" \
                        + "%s_Features/%s_%s_%s.mat" % (model_name, model_name, layer, suffix)
            mat_dict1 = dict(data = data1, data_shape = data1.shape)           
            scipy.io.savemat(mat_outpath1, mat_dict1)       
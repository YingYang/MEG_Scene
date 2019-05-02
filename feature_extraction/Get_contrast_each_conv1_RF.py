"""
Given the conv1 layer of 96x55x55 in AlexNet structure,
reconstruct 3x227x227 images, using the filters
# The filter operation is not totally correct, so this code can not be used. 
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.misc 
im_width = 227
stride = 4
filter_width = 11
half_filter_width = 5

# use least square to reconstruct 5x5 filters
n_stride = 10
im_patch_width = 11+stride*(n_stride-1)
n_rf = 55


# get the contrast of each RF in Layer 1, n_im x 55 x 55
# then regress out the aspect things
# save the mat files


isExtra_image = False

if isExtra_image:
    image_path_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Extra_Images/Extra_Images_list.txt"
    input_image_seq = np.loadtxt(image_path_fname, delimiter = " ", dtype = np.str_)
    input_image_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Extra_Images/Extra_Images_boxed/"
else:
    image_path_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Images/Image_List.txt"
    input_image_seq = np.loadtxt(image_path_fname, delimiter = " ", dtype = np.str_)
    input_image_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Images/"





# load the test images
n_im = len(input_image_seq)
contrast = np.zeros([n_im, n_rf, n_rf])
for j in range(n_im):
    image_path = input_image_dir + input_image_seq[j]
    tmp_image = scipy.misc.imread(image_path)
    tmp_image = scipy.misc.imresize(tmp_image, [im_width, im_width])    
    # convert to gray value
    tmp_image_gray = tmp_image.astype(np.float).mean(axis = 2)
    for i1 in range(n_rf):
        for i2 in range(n_rf):
            center1, center2 = i1*4+half_filter_width, i2*4+half_filter_width
            tmp_patch = tmp_image_gray[center1-half_filter_width:center1+half_filter_width+1,
                                       center2-half_filter_width:center2+half_filter_width+1]
            contrast[j, i1,i2] = (tmp_patch.max()-tmp_patch.min()) \
                                 /(tmp_patch.max()+tmp_patch.min())
    


contrast[np.isnan(contrast)] = 0.0
contrast = contrast.reshape([n_im, -1])


mat_path = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/"
if isExtra_image:
    mat_name =  mat_path+"Extra_images_Layer1_contrast.mat"
else:
    mat_name = mat_path+"Stim_images_Layer1_contrast.mat"

scipy.io.savemat(mat_name, dict(contrast = contrast))




#%%==================== remove the aspect for stim images only ===============

if not isExtra_image:
    #cm = plt.get_cmap("gray")
    #plt.figure()
    #plt.subplot(2,1,1)
    #plt.imshow(contrast[j], cmap = cm)
    #plt.subplot(2,1,2)
    #plt.imshow(tmp_image_gray, cmap = cm)
    
    
    
    # regress out the aspect 
    
    mat_dict = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/aspect_energy.mat")
    X = mat_dict['X0']
    #projector = np.eye(n_im)- X.dot(np.linalg.inv(X.T.dot(X)).dot(X.T))
    proj =  mat_dict['proj'] 
    
    # proj included an extra all-one column to subtract the mean
    #print np.linalg.norm(projector-proj)/np.linalg.norm(proj)
    
    contrast_noaspect = np.dot(proj, contrast)
    mat_path = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/Stim_images_Layer1_contrast_noaspect.mat"
    scipy.io.savemat(mat_path, dict(contrast_noaspect = contrast_noaspect))
    u,d,v = np.linalg.svd(contrast_noaspect)
    a = np.cumsum(d**2)/np.sum(d**2)
    plt.plot(a,'-*')
    plt.grid('on')
    
#==================== remove the aspect of both stim and extra
if False:
    import scipy.stats
    
    mat_path = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/"
    mat_name1 = mat_path+"Stim_images_Layer1_contrast.mat"
    mat_name2 =  mat_path+"Extra_images_Layer1_contrast.mat"
    
    contrast1 = scipy.io.loadmat(mat_name1)['contrast']
    contrast2 = scipy.io.loadmat(mat_name2)['contrast']
    contrast = np.vstack([contrast1, contrast2])
    
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
    
    contrast_noaspect = np.dot(proj, contrast)
    
    mat_name3 = mat_path+"All_images_Layer1_contrast.mat"
    mat_dict3 = dict(contrast_noaspect = contrast_noaspect)
    scipy.io.savemat(mat_name3, mat_dict3)
    

"""
# I tried to reconstruct the image from the Layer1 response, by superposing the
# filters, but for some reason it did not work (20160921)

n_filters = 96

filters = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/AlexNet_conv1_filters.mat")  
filters = filters['filters']       
       
# for each patch, compute the inverse operator     
       
# vec(im_patch)*W = vec(filter_res),
# vec(im_patch) = vec(filter_res) W.T (W W^T)^-1
feature_width = n_stride
W = np.zeros([n_filters, feature_width, feature_width, 
              3, im_patch_width, im_patch_width]) 
     
for l in range(n_filters):        
    for i in range(n_stride):
        for j in range(n_stride):
            tmp = np.zeros([3, im_patch_width, im_patch_width])
            tmp[:,(5+i*stride-half_filter_width):(6+i*stride+half_filter_width),\
                (5+j*stride-half_filter_width):(6+j*stride+half_filter_width)] = filters[l]
            W[l,i,j] = tmp
 
W_mat = W.reshape([n_filters* feature_width* feature_width, 
              3*im_patch_width* im_patch_width]).T  
# inverse operator
# inv = W^T(WWT)^{-1}              
inv = W_mat.T.dot(np.linalg.inv(W_mat.dot(W_mat.T)))

#========== given the feature 96x55x55 of one image, reconstruct patch by patch

#conv1_feat = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/"+
#           "Features/AlexNet_Features/AlexNet_Extra_Images_conv1.mat")
#
tmp_dict = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/"+\
           "AlexNet_Features/AlexNet_Extra_Image_Features/AlexNet_Image0.mat")            
conv1_feat = tmp_dict['conv1']
del(tmp_dict)
tmp_feat = conv1_feat           
tmp_im = np.zeros([3,227,227])
mask = np.zeros([3,227,227])

n_large_stride = 55/n_stride
large_stride = 20
half_patch_width = im_patch_width//2
for i in range(n_large_stride):
    for j in range(n_large_stride):
        mask[:,(13+i*large_stride-half_patch_width):(14+i*large_stride+half_patch_width),
               (13+j*large_stride-half_patch_width):(14+j*large_stride+half_patch_width)]+= 1


for i in range(n_large_stride):
    for j in range(n_large_stride):
        tmp_feat_patch = tmp_feat[:,i*n_stride:(i+1)*n_stride, j*n_stride:(j+1)*n_stride]
        tmp_im_patch = tmp_feat_patch.ravel().dot(inv)
        tmp_a = tmp_im_patch.reshape([3, im_patch_width, im_patch_width])
        tmp_im[:,(13+i*large_stride-half_patch_width):(14+i*large_stride+half_patch_width),
               (13+j*large_stride-half_patch_width):(14+j*large_stride+half_patch_width)]\
               = tmp_a
tmp_im = tmp_im/mask


# note the mean of the image must be subtacted
# Conv1 is after relu
# Wrong: after relu, there is no way of reconstruct it
# The scales are different, weird. 

#=============== debuging load the image too
image_path = "/home/ying/Dropbox/Scene_MEG_EEG/Extra_Images/Extra_Images_boxed/Im362_air_base_1.jpg"
im = scipy.misc.imread(image_path)
im = scipy.misc.imresize(im,[227,227,3])
im = np.transpose(im/255.0,[2,0,1])
#im = np.transpose(im,[2,0,1])


caffe_root = '/home/ying/Packages/caffe/caffe-master/'
default_mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
mean_matrix = np.load(default_mean_path)
im_mean = mean_matrix.transpose([1,2,0])
im_mean = scipy.misc.imresize(im_mean,[227,227,3])/255.0
im_mean = im_mean.transpose([2,0,1])

im = im-im_mean
im1 = im.copy()
# Image musts be BGR
im1[2] = im[0].copy()
im1[0] = im[2].copy()

i,j = 6,9
a = im1[:,(13+i*large_stride-half_patch_width):(14+i*large_stride+half_patch_width),
               (13+j*large_stride-half_patch_width):(14+j*large_stride+half_patch_width)]
b = tmp_feat[:,i*n_stride:(i+1)*n_stride, j*n_stride:(j+1)*n_stride]

plt.figure(); plt.imshow(b[0], interpolation = "none", vmin = 0); plt.colorbar()
c = a.ravel().dot(W_mat)
c2 = c.reshape([96,5,5])
plt.figure(); plt.imshow(c2[0], interpolation = "none", vmin = 0); plt.colorbar()
c3 = (a*W[0,0,0]).sum()

if False:
    plt.figure()
    for i in range(96):
        _= plt.subplot(10,10,i+1)
        _= plt.imshow(tmp_feat[i])
        
"""        
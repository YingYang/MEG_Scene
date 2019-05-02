import numpy as np
import scipy.io
import matplotlib.pyplot as plt


if True: # used images
    input_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Images/"
    image_path_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Images/Image_List.txt"
    mat_data = scipy.io.loadmat('/home/ying/Dropbox/Scene_MEG_EEG/Features/selected_image_second_round_data.mat');
    out_mat_name = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/aspect_energy.mat"
else:  # extra images
    input_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Extra_Images/Extra_Images_boxed/"
    image_path_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Extra_Images/Extra_Images_list.txt"
    mat_data = scipy.io.loadmat('/home/ying/Dropbox/Scene_MEG_EEG/Extra_Images/Extra_Images.mat');
    out_mat_name = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/aspect_energy_extra_images.mat"
    
    
#============= im size, area, aspect ratio and total energy =================== 
image_path_seq = np.loadtxt(image_path_fname, delimiter = " ", 
                                 dtype = np.str_)
n_im = len(image_path_seq)                                
aspect_ratio = mat_data['aspect_ratio'] [:,0]
max_side = 500.0
full_side = 600.0
gray_box = np.ones([n_im, 600,600])
im_size = np.zeros([n_im, 2])
# im_size = width, hight
for i in range(n_im):
    if aspect_ratio[i] >= 1:
        im_size[i,0] = max_side
        im_size[i,1] = max_side/aspect_ratio[i]
    else:
        im_size[i,0] = max_side*aspect_ratio[i]
        im_size[i,1] = max_side
    half_width = im_size[i,:]//2       
    gray_box[i, (full_side)/2-half_width[1]:(full_side)/2+half_width[1],
             (full_side)/2-half_width[0]:(full_side)/2+half_width[0]] = 0.0
                 
mean_gray = np.zeros(n_im)
for i in range(n_im):
    image_path = input_dir + image_path_seq[i]
    tmp_im = scipy.misc.imread(image_path)
    tmp1 = np.mean(tmp_im, axis = -1)
    mean_gray[i] = np.mean(tmp1[gray_box[i] ==0])
   
X = np.zeros([n_im, 5])
# first row is all one 

X[:,0:2] = im_size
X[:,2] = im_size[:,0]*im_size[:,1]
X[:,3] = im_size[:,0]/im_size[:,1]
X[:,4] = mean_gray
import scipy.stats
X1= scipy.stats.zscore(X, axis = 0)  
# do an SVD of X
u,d,v = np.linalg.svd(X1, full_matrices = False)
X_reg = np.ones([n_im,6])
X_reg[:,0:5] = u

# projection matrix  I - X(XTX)^{-1}XT
inv = np.linalg.inv(X_reg.T.dot(X_reg))
proj = np.eye(n_im) - (X_reg.dot(inv)).dot(X_reg.T)

mat_dict = dict(X = X_reg, X0 = X, proj = proj,
                X_col_names = ["width","height","area","aspectratio","mean_gray","one"])
scipy.io.savemat(out_mat_name, mat_dict)

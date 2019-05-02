import numpy as np
import scipy.io 
import matplotlib.pyplot as plt

# Layer 1 6
#mode_id = 1
# local contrast and Layer 6
mode_id = 5

mode_names = ['','Layer1_6','localcontrast_Layer6', 'Layer1_7_noncontrast160',
              'Layer1_6_noncontrast160', 'Layer1_7']

if mode_id == 1:
    layers = ['conv1','fc6']
elif mode_id == 2:
    layers = ['localcontrast','fc6']
elif mode_id == 3:
    layers = ['conv1_nc160','fc7_nc160']
elif mode_id == 4:
    layers = ['conv1_nc160','fc6_nc160']
elif mode_id == 5:
    layers = ['conv1','fc7']
    
# load the data
    
dropbox_dir = "/media/yy/LinuxData/yy_dropbox/Dropbox/"
mat_dict = scipy.io.loadmat(dropbox_dir + "/Scene_MEG_EEG/Features"+
          "/CCA_%s_%s_CCA_test.mat" %(layers[0], layers[1]))
# UD of the two layers          
Xtest = mat_dict['Xtest']
Proj_test = mat_dict['Proj_test']

'''
# plot the correlation of the test
n0 = 100
corr = np.zeros(n0)
for i in range(n0):
    tmp_corr = np.corrcoef(Proj_test[0,1][:,i], Proj_test[0,0][:,i])
    corr[i] = np.abs(tmp_corr[0,1])
    
plt.figure(figsize = (12,3))
plt.plot(np.arange(1,n0+1),corr,'-+')
plt.xlabel('index of components')
plt.ylabel('abs correlation of \n the projections')    
plt.tight_layout()
plt.grid()
fig_out_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Features/"
plt.savefig(fig_out_dir + "Features_%s_CCA_test_corr.eps" % mode_names[mode_id])
 '''   


# demean
for i in range(2):
    Xtest[0,i] -= np.mean(Xtest[0,i], axis = 0)
    plt.figure(); plt.plot(Xtest[0,i].mean(axis = 0))
n_im = Xtest[0,1].shape[0]


save_flag = True
if save_flag:
    #outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
    # for HBM revision, Reviewer 2 question
    outdir = "/media/yy/LinuxData/yy_Scene_MEG_data/regressor/"


    n_CC = 6
    #n_CC = 3
    
    # how to compute residuals? take the common part out?
    # or take the closest projections to the common part out seperately?
    # option 1, merge the data, reguress out the first n_CC component of the merge
    merged_Proj = np.hstack([Proj_test[0,0][:,0:n_CC], Proj_test[0,1][:,0:n_CC]])
    merged_Proj -= np.mean(merged_Proj, axis = 0)
    merged_u, merged_d, merged_v = np.linalg.svd(merged_Proj)
    merged_u1 = merged_u[:,0:n_CC].copy()
    merged_u1_aug = np.hstack([merged_u1, np.ones([n_im,1])])
    # regression operator I - X(X^T X)^{-1} X^T
    Proj_operator = np.eye(n_im)-  reduce( np.dot, 
                            [merged_u1_aug,  np.linalg.inv(merged_u1_aug.T.dot(merged_u1_aug)), merged_u1_aug.T])
    for i in range(2):
        tmp_data = Proj_operator.dot(Xtest[0,i])
        # svd
        tmpu, tmpd, tmpv = np.linalg.svd(tmp_data)
        plt.figure();plt.plot(tmpu.mean(axis = 0))
        mat_name = outdir+ "AlexNet_%s_%s_cca_%d_residual_svd.mat" %(mode_names[mode_id], layers[i], n_CC)
        scipy.io.savemat(mat_name, dict(u = tmpu, d = tmpd, v=tmpv, 
                                     n_CC = n_CC, 
                                     merged_u = merged_u, merged_d = merged_d, merged_v = merged_v))
    
    
    # option 2, regress out the projections sepereately
    for i in range(2):
        # regression operator I - X(X^T X)^{-1} X^T
        tmp_proj = np.hstack([Proj_test[0,i][:,0:n_CC], np.ones([n_im,1])])
        Proj_operator = np.eye(n_im)- reduce(np.dot, [tmp_proj, np.linalg.inv(tmp_proj.T.dot(tmp_proj)), tmp_proj.T])
        tmp_data = Proj_operator.dot(Xtest[0,i])
        # svd
        tmpu, tmpd, tmpv = np.linalg.svd(tmp_data)
        plt.figure(); plt.plot(tmpu.mean(axis = 0))
        mat_name = outdir+ "AlexNet_%s_%s_cca_%d_sep_residual_svd.mat" %(mode_names[mode_id], layers[i], n_CC)
        scipy.io.savemat(mat_name, dict(u = tmpu, d = tmpd, v=tmpv))
        


#%%
# compute variance explained by the extracted features using ols
# load the features of the 362 images


regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
feature_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Features/"

n_comp = 3
n_im = 362
layer0 = ['conv1','fc7']

raw_feature_list = []
for i in range(len(layer0)):
    raw_feature_mat_name = feature_dir+\
              'AlexNet_Features/AlexNet_%s_noaspect_all_images.mat'%layer0[i]
    raw_feature = scipy.io.loadmat(raw_feature_mat_name)['data'][0:n_im]
    raw_feature -= np.mean(raw_feature)
    raw_feature_list.append(raw_feature)
    
# add Layer 1 contrast
mat_path = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/Stim_images_Layer1_contrast_noaspect.mat"
#mat_path = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/Stim_images_Layer1_contrast.mat"
mat_dict = scipy.io.loadmat(mat_path)
raw_feature_list.append(mat_dict['contrast_noaspect']-np.mean(mat_dict['contrast_noaspect']))
#raw_feature_list.append(mat_dict['contrast']-np.mean(mat_dict['contrast']))

# add the features without the contrast
for i in range(len(layer0)):
    raw_feature_mat_name = regressor_dir+ "AlexNet_%s_no_aspect_no_contrast160_all_im_PCA.mat" %(layer0[i])
    raw_feature = scipy.io.loadmat(raw_feature_mat_name)['data'][0:n_im]
    raw_feature -= np.mean(raw_feature)
    raw_feature_list.append(raw_feature)


X_list = []
for i in range(2):
    tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_%s_cca_%d_residual_svd.mat" %(mode_names[mode_id], layers[i], n_comp))
    X_list.append(tmp['u'][:,0:n_comp])
    X_list.append(tmp['merged_u'][:,0:n_comp])
    
# only looke at the first a few components
    
for i in range(len(X_list)):
    X_list[i] = X_list[i][:,0:10]
 

raw_feature_name = layer0 + ['contrast_noaspect'] + layer0
X_list_name = ['resi1','common','resi7','common']   
   
for i in range(len(raw_feature_list)): 
    # compute average variance explained
    raw_feature = raw_feature_list[i]
    for l in range(len(X_list)):
        tmp_X = X_list[l] 
        tmp_X -= np.mean(tmp_X)
        proj = tmp_X.dot(np.linalg.inv(tmp_X.T.dot(tmp_X))).dot(tmp_X.T)
        residual = raw_feature - proj.dot(raw_feature)
        prop_var = 1- np.sum(residual**2)/np.sum(raw_feature**2)
        print raw_feature_name[i], X_list_name[l], np.round(prop_var, decimals = 3)*100.0




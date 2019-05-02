import numpy as np
import scipy.io
import scipy.stats
import matplotlib.pyplot as plt


outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/ridge_reg/MEG/"

feat_name_seq = ["neil_attr","neil_low","neil_scene","conv1","conv2","conv3","conv4","conv5","fc6","fc7"]
n_feat_name = len(feat_name_seq)


Subj_list = np.arange(1, 10)
n_subj = len(Subj_list)
error_ratio_all = np.zeros([n_subj, n_feat_name, 306,110])

#suffix = "16_dim"
suffix = "80_percent"

for i in range(n_subj):
    subj = "Subj%d" %Subj_list[i]
    for j in range(len(feat_name_seq)):
        feat_name = feat_name_seq[j]
        mat_name = outdir + "%s_1_50Hz_raw_ica_window_50ms_%s_%s_all_trials.mat" %(subj, feat_name, suffix)
        mat = scipy.io.loadmat(mat_name)
        tmp = np.zeros([306,110])
        tmp[mat['picks_all'][0],:] = mat['error_ratio']
        error_ratio_all[i,j] = tmp
        
    
    
plt.figure()
m1,m2 = 3,4
vmin, vmax = -0.5,0.5
for j in range(n_feat_name):
    plt.subplot(m1,m2,j+1)
    plt.imshow(-np.log10(error_ratio_all[:,j,:,:]).mean(axis = 0),
               aspect = "auto", interpolation = "none", vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title(feat_name_seq[j])



vmin, vmax = -6, 6
plt.figure()   
diff = -np.log10(error_ratio_all[:,0]) + np.log10(error_ratio_all[:,7])
diff_T = np.mean(diff, axis = 0)/np.std(diff, axis = 0)* np.sqrt(n_subj)
plt.imshow(diff_T, vmin = vmin, vmax = vmax, interpolation = "None", aspect = "auto")
plt.colorbar()

plt.figure()
vmin, vmax = None, None
m1,m2 = 3,3
for i in range(n_subj):
    plt.subplot(m1,m2,i+1)
    plt.imshow(diff[i], aspect = "auto", interpolation = "none", vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title("Subj%d" %Subj_list[i])

    
#========================== do some test to check why conv1 and 
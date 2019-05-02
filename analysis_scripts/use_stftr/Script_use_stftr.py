"""
Usage: python Script_use_stftr.py  Subj1
"""
import scipy.io
import numpy as np
import mne
from copy import deepcopy
import pickle
import sys
import getpass
username = getpass.getuser()
Flag_on_cluster = True if username == "yingyan1" else False
if Flag_on_cluster:
    paths = ["/home/yingyan1/Scene_MEG_EEG/analysis_python_code/use_stftr/"]
else:
    paths = ["/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/use_stftr/"]
for l in range(len(paths)):
    sys.path.insert(0,paths[l])
from use_stftr import stftr_on_feat_L21, stftr_on_feat_L2

MEGorEEG = ["EEG","MEG"]
MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect"
 
subj = str(sys.argv[1])
modality = str(sys.argv[2])
isMEG = True if modality == "MEG" else False
print "subj = %s, isMEG %d" %(subj, isMEG)
fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
print "fname_suffix"
print fname_suffix

# usage python Script_use_stftr.py Subj1
if Flag_on_cluster:
    datadir = "/data2/tarrlab/MEG_NEIL/"
    if isMEG:
        fwd_path = datadir  + "MEG_preprocessed_data/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
        epochs_path = (datadir + "MEG_preprocessed_data/epoch_raw_data/%s/%s_%s" \
                    %(subj, subj, "run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"))
        datapath = datadir + "MEG_preprocessed_data/epoch_raw_data/%s/%s_%s.mat" \
                  %(subj, subj, fname_suffix)
    else:
        fwd_path = datadir + "EEG_preprocessed_data/fwd/%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)
        #epochs_path = ( datadir +"EEG_preprocessed_data/epoch_raw_data/%s_EEG/%s_EEG_%s" \
        #            %(subj, subj, "filter_1_110Hz_notch_ica_reref_smoothed-epo.fif.gz"))
        epochs_path = ( datadir +"EEG_preprocessed_data/epoch_raw_data/%s_EEG/%s_EEG_%s" \
                    %(subj, subj, "filter_1_110Hz_notch_PPO10POO10_swapped_ica_reref_smoothed-epo.fif.gz"))
        datapath = datadir + "EEG_preprocessed_data/epoch_raw_data/%s_EEG/%s_EEG_%s.mat" \
                  %(subj, subj, fname_suffix)
    stc_outdir = "/data2/tarrlab/MEG_NEIL/MEG_EEG_results/%s_stftr/" %(MEGorEEG[isMEG])
    regressor_dir = "/data2/tarrlab/MEG_NEIL/regressor/"   
else:
    datadir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"
    if isMEG:
        fwd_path = datadir  + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
        epochs_path = (datadir + "MEG_DATA/DATA/epoch_raw_data/%s/%s_%s" \
                    %(subj, subj, "run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"))
        datapath = datadir + "MEG_DATA/DATA/epoch_raw_data/%s/%s_%s.mat" \
                  %(subj, subj, fname_suffix)
    else:
        fwd_path = datadir + "EEG_DATA/DATA/fwd/%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)
        epochs_path = ( datadir +"EEG_DATA/DATA/epoch_raw_data/%s_EEG/%s_EEG_%s" \
                    %(subj, subj, "filter_1_110Hz_notch_ica_reref_smoothed-epo.fif.gz"))
        datapath = datadir + "EEG_DATA/DATA/epoch_raw_data/%s_EEG/%s_EEG_%s.mat" \
                  %(subj, subj, fname_suffix)
    stc_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/%s/" %(MEGorEEG[isMEG])
    regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"



# originally using 4 components with contrast, the alpha,beta,gamma 27,40,14
# needs to be modifiled:    
alpha_seq = np.exp(np.arange(3,1,-1))*10.0
beta_seq = np.exp(np.arange(3,1,-1))*20.0
gamma_seq = np.exp(np.arange(1,0,-1))*3.0
delta_seq = np.exp(np.arange(-6,6,2))
#delta_seq = np.array([1E-1])
snr_tuning_seq = [0.5, 1.0, 2.0, 3.0]

# debug
#alpha_seq = alpha_seq[-1::]
#beta_seq = beta_seq[-1::]
#gamma_seq = gamma_seq[-1::]
print alpha_seq
print beta_seq
print gamma_seq

#alpha_seq = np.exp(np.arange(1,-2,-1))*10.0
#beta_seq = np.exp(np.arange(3,-2,-1))*2.0
#gamma_seq = np.exp(np.arange(2,-2,-1))*2.0


"""
n_comp = 4
feat_name = ['conv1','fc6']
X_list = list()
for j in range(2):
    tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_cca_%d_residual_svd.mat" %(feat_name[j], n_comp))
    X_list.append(tmp['u'][:,0:n_comp])

X_list.append(tmp['merged_u'][:,0:n_comp])
feat_name.append( "CCA_merged")
n_sub_feat = len(X_list)
pairs = [[1,0]]  
n_pair = len(pairs) 
pair_names = ['Layer6-Layer1']
X_name = feat_name

mat_name = stc_outdir + "train_test_id_boostrap_seq.mat"
# check if it is there
mat_dict = scipy.io.loadmat(mat_name)
train_im_id = mat_dict['train_im_id'][0]
test_im_id = mat_dict['test_im_id'][0]
bootstrap_seq = mat_dict['bootstrap_seq'].astype(np.int)
bootstrap_seq = bootstrap_seq[0:B]
print bootstrap_seq.shape
"""
CCA_flag = True


if not CCA_flag:
    n_comp = 10
    feature_suffix = "no_aspect_no_contrast100"
    model_name = "AlexNet"
    #feat_name_seq = ["rawpixelgraybox"]
    feat_name_seq = []
    layers = ["conv1",  "fc7"]
    #feat_name_seq = list()
    for l in layers:
        feat_name_seq.append("%s_%s_%s" %(model_name,l,feature_suffix))
    n_feat = len(feat_name_seq) 
                
    X_list = []
    for j in range(n_feat):  
        # load the design matrix 
        feat_name = feat_name_seq[j]
        print feat_name
        regressor_fname = regressor_dir +  "%s_PCA.mat" %(feat_name)
        tmp = scipy.io.loadmat(regressor_fname)
        X0 = tmp['X']
        X = X0[:,0:n_comp]
        X_list.append(X)

    fname = "/home/ying/Dropbox/Scene_MEG_EEG/Features/Layer1_contrast/Stim_images_Layer1_contrast_noaspect.mat"
    mat_dict = scipy.io.loadmat(fname)
    tmp = mat_dict['contrast_noaspect']
    tmp = tmp- np.mean(tmp, axis = 0)
    U,D,V = np.linalg.svd(tmp)
    var_prop = np.cumsum(D**2)/np.sum(D**2)
            
    X_list.append(U[:,0:n_comp])
    feat_name_seq.append("contrast")
    n_feat = len(X_list) 
    
    sol_path = stc_outdir + "%s_STFT-R_all_image_Layer_1_7_con_ncomp%d_%s" % (subj, n_comp, MEGorEEG[isMEG])  

else:
    n_comp1 = 6
    n_comp = 6
    #feat_name_seq = ['conv1','fc6']
    feat_name_seq = ['conv1_nc160','fc7_nc160']
    mode_id = 3
    mode_names = ['','Layer1_6','localcontrast_Layer6', 'Layer1_7_noncontrast160']
    
    X_list = list()
    for j in range(2):
        #tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_cca_%d_residual_svd.mat" %(feat_name_seq[j], n_comp1))
        mat_name = regressor_dir+ "AlexNet_%s_%s_cca_%d_residual_svd.mat" %(mode_names[mode_id], feat_name_seq[j], n_comp1)
        tmp = scipy.io.loadmat(mat_name)
        X_list.append(tmp['u'][:,0:n_comp])
    
    X_list.append(tmp['merged_u'][:,0:n_comp])
    feat_name_seq.append( "CCA_merged")
    
    sol_path = stc_outdir + "%s_STFT-R_all_image_Layer_1_7_CCA_ncomp%d_%s" % (subj, n_comp, MEGorEEG[isMEG])  


X = np.hstack([X_list[0],X_list[1]])
X = np.hstack([X,X_list[2]])

"""
import matplotlib.pyplot as plt
plt.imshow(np.abs(np.corrcoef(X.T))); plt.colorbar();
"""

print sol_path            
n_im = 362
train_im_id = range(0, n_im)
test_im_id = None


#============================================================
wsize = 16
tstep = 4
flag_L21 = True
if flag_L21:
    print datapath
    print epochs_path
    stftr_on_feat_L21(subj, X, fwd_path, epochs_path, datapath, sol_path,
                     train_im_id, test_im_id,
                     alpha_seq, beta_seq, gamma_seq,
                     isMEG = True, t_cov_baseline = -0.05,
                     wsize = wsize, tstep = tstep, 
                     maxit = 200, tol = 5E-4, Incre_Group_Numb = 1000, Maxit_J = 8, dual_tol = 0.1,
                     Flag_verbose = False, depth = 0.8,
                     label_names = None, n_active_ini = 1000)
"""
else: 
    sol_outdir = stc_outdir
    stftr_on_feat_L2(subj, fwd_path, sol_path, sol_outdir, bootstrap_seq, 
                     delta_seq, snr_tuning_seq = None,
                     method = "STFT-R",
                     isMEG = True, t_cov_baseline = -0.05,
                     wsize = wsize, tstep = tstep, 
                     maxit = 200, tol = 1E-3, 
                     Flag_verbose = False, depth = 0.8,
                     label_names = None)

"""
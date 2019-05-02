import numpy as np
import scipy.io
import mne
mne.set_log_level('WARNING')
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import scipy.io
import scipy.spatial
import sys
#path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/"

sys.path.insert(0, path0)
#path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
path1 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
#
sys.path.insert(0, path1)
from ols_regression import ols_regression
#path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
#sys.path.insert(0, path0)

dropbox_path = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/"


mat_dict = scipy.io.loadmat(dropbox_path +"/Features/CCA_conv1_fc7_SVD_UD.mat")

data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
data_root_dir =data_root_dir0 +"MEG_preprocessed_data/MEG_preprocessed_data/"

#=====================================================
ROI_bihemi_names = ['pericalcarine', 'PPA_c_g', 'TOS_c_g', 'RSC_c_g', 'LOC_c_g']                
labeldir = data_root_dir + "/ROI_labels/"  
MEGorEEG = ['EEG','MEG']

#%% Some initial parameters to set
isMEG = True
#Flag_CCA = False
Flag_CCA = True
lambda2 = 1

#%%
if True:
#for isMEG in [False, True]:
    #
    if isMEG:
        subj_list = range(1, 19)
        times = np.arange(0.01,1.01,0.01)
        n_times = 100
        time_in_ms = (times - 0.04)*1000.0
    else:
        subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
        times = np.arange(0.01,1.0,0.01)
        n_times = 99
        time_in_ms = times*1000.0

    # For now for MEG only
    stc_out_dir = data_root_dir0 + "Result_Mat/" \
                + "source_solution/dSPM_%s_ave_per_im/" % \
                (MEGorEEG[isMEG])
    
    flag_swap_PPO10_POO10 = True
    MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect" \
        if flag_swap_PPO10_POO10 else "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"  
    fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
        
    
    n_subj = len(subj_list)
    n_im = 362
    regressor_dir = data_root_dir0 + "/regressor/"
    #offset = 0.04 if isMEG else 0.02
      
     
    if Flag_CCA:
        n_comp1 = 6
        n_comp = 6
        feat_name_seq = ['conv1','fc7']
        #feat_name_seq = ['conv1_nc160','fc7_nc160']
        mode_id = 4
        mode_names = ['','Layer1_6','localcontrast_Layer6', 'Layer1_7_noncontrast160',
                      'Layer1_7']
        X_list = list()
        for j in range(2):
            #tmp = scipy.io.loadmat(regressor_dir + "AlexNet_%s_cca_%d_residual_svd.mat" %(feat_name_seq[j], n_comp1))
            mat_name = regressor_dir+ "AlexNet_%s_%s_cca_%d_residual_svd.mat" %(mode_names[mode_id], feat_name_seq[j], n_comp1)
            tmp = scipy.io.loadmat(mat_name)
            X_list.append(tmp['u'][:,0:n_comp])
        
        X_list.append(tmp['merged_u'][:,0:n_comp])
        feat_name_seq.append( "CCA_merged") 
        pairs = [[1,0],[0,2],[1,2]]  
        n_pair = len(pairs) 
        pair_names = ['Layer7-Layer1','Layer1-Common','Layer7-Common']
        n_feat = len(X_list) 
    else:   
        n_comp = 10
        #feature_suffix = "no_aspect"
        # 100 dimensions of the local contrast regressed out. 
        feature_suffix = "no_aspect_no_contrast160_all_im" 
        model_name = "AlexNet"
        #feat_name_seq = ["rawpixelgraybox"]
        feat_name_seq = []
        if True:
            layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", ] # "prob"]
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
            X0 = tmp['X'][0:n_im]   
            suffix = "%d_dim" % n_comp
            print suffix, n_comp
            X = X0[:,0:n_comp]
            X_list.append(X) 
            
        pairs = []  
        n_pair =0
        pair_names = []
    
    
    ROI_partition_dir = data_root_dir + "ROI_labels/" 
    n_part = 3
    quantile_flag = False
    nROI0 = len(ROI_bihemi_names)
    for l in range(n_part):
        ROI_bihemi_names.append("vent_%d_qtl%d" %(l, quantile_flag))  
    nROI = len(ROI_bihemi_names)    
    #================== actual regression =========================  
    n_reg = len(X_list)
    val = np.zeros([n_subj,nROI,n_reg, n_times])
    val_normalized = np.zeros([n_subj,nROI,n_reg, n_times])
    val_normalized_timewise = np.zeros([n_subj,nROI,n_reg, n_times])
    
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        labeldir1 = labeldir + "%s/labels/" % subj
        # load and merge the labels
        labels_bihemi = list()
        for j in range(nROI0):
            tmp_name = ROI_bihemi_names[j]
            tmp_label_list = list()
            for hemi in ['lh','rh']:
                print subj, tmp_name, hemi
                tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,tmp_name,hemi)
                tmp_label = mne.read_label(tmp_label_path)
                tmp_label_list.append(tmp_label)
            labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1]) 
        if isMEG:
            fwd_path = data_root_dir + "/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
        else:
            fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                               + "EEG_DATA/DATA/fwd/%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)
            # TBC
            
            
        fwd = mne.read_forward_solution(fwd_path) 
        fwd = mne.convert_forward_solution(fwd, surf_ori = True,)
        src = fwd['src']
        ROI_ind = list()
        for j in range(nROI0):
            tmp_label = labels_bihemi[j]
            _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
            ROI_ind.append(tmp_src_sel)
       
        # add ROI indices
        stc_name = ROI_partition_dir + "%s/labels/%s_ventral_divide%d_label_quantile%d" %(subj, subj, n_part, quantile_flag)
        tmp_stc = mne.source_estimate.read_source_estimate(stc_name)
        dipole_label_val = tmp_stc.data[:,0]
        for i0 in range(n_part):
            tmp_ind = np.all( np.vstack([ dipole_label_val >= i0+0.5, dipole_label_val < i0+1.5]), axis = 0)
            ROI_ind.append(np.nonzero(tmp_ind)[0])
        del(stc_name, tmp_stc, dipole_label_val)    
            
        # load the source solution
        mat_dir = stc_out_dir + "%s_%s_%s_lambda2_%1.1f_ave.mat" %(
                subj, MEGorEEG[isMEG],fname_suffix, lambda2)
        mat_dict = scipy.io.loadmat(mat_dir)
        source_data = mat_dict['source_data']
        times = mat_dict['times'][0]
        del(mat_dict)
            
        ROI_data = list()
        for j in range(nROI):
            tmp = source_data[:,ROI_ind[j],:]
            tmp -= np.mean(tmp, axis = 0)
            ROI_data.append(tmp)
        del(source_data) 
        
        for j in range(nROI):
            tmp_val = np.zeros([len(ROI_ind[j]), n_reg, n_times])
            for j1 in range(n_reg):
                tmpX = X_list[j1]
                tmpX -= np.mean(tmpX, axis = 0)
                tmp_result = ols_regression(ROI_data[j], tmpX)
                #tmp_val[:,j1,:] = tmp_result['log10p']
                tmp_val[:,j1,:] = tmp_result['Rsq'][:,0:n_times]
            val[i,j] = np.mean(tmp_val,axis = 0)  
            val_normalized[i,j,:] = (tmp_val.transpose([1,2,0]) \
                   / np.sum(np.sum(tmp_val, axis = -1),axis = -1)).mean(axis = -1)
            val_normalized_timewise[i,j,:] = (tmp_val.transpose([1,0,2]) \
                   / np.sum(tmp_val, axis = 1)).mean(axis = 1) 
     
    mat_dict = dict(val = val, #diff_val = diff_val,
                    val_normalized = val_normalized,
                    val_normalized_timewise = val_normalized_timewise,
                    ROI_bihemi_names = ROI_bihemi_names,
                    feat_name_seq = feat_name_seq, pairs = pairs, pair_names = pair_names,
                    times = times,)
    mat_outdir =data_root_dir0 + "/Result_Mat/source_regression/dSPM_reg_ROI/"        
    mat_name =  mat_outdir  + "%s_dSPM_ROI_lambda2_%1.1f_ols_reg_CCA%d_%d_modeName%s.mat" \
                %(MEGorEEG[isMEG], lambda2, Flag_CCA, n_comp, mode_names[mode_id])
    scipy.io.savemat(mat_name, mat_dict)   




#================ load the results 
if False:
    isMEG = True
    MEGorEEG = ["EEG","MEG"]
    n_comp = 10
    mat_outdir = data_root_dir0 + "Result_MAT/source_regression/dSPM_reg_ROI"        
    mat_name =  mat_outdir  + "%s_dSPM_ROI_ols_reg_CCA%d.mat" %(MEGorEEG[isMEG],n_comp)
    mat_dict = scipy.io.loadmat(mat_name)
    
    val = mat_dict['val']
    
    diff_val = np.zeros([n_subj,nROI,n_pair,n_times])
    for l in range(n_pair):
        diff_val[:,:,l,:] = val[:,:,pairs[l][0]] - val[:,:,pairs[l][1]]
    ROI_id = 0
    val = mat_dict['val']
    times = mat_dict['times'][0]
    # plot each single subject
    plt.figure()
    for i in range(val.shape[0]):
        _=plt.subplot(4,4,i+1)
        _=plt.plot(times, val[i,ROI_id].T)
        
    
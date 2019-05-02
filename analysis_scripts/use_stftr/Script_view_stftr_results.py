"""
Only run this locally on m tarrlabb434.psy.cmu.edu
"""
import scipy.io
import numpy as np
import mne
from copy import deepcopy
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import getpass
username = getpass.getuser()
Flag_on_cluster = True if username == "yingyan1" else False
paths = ["/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/use_stftr/"]
for l in range(len(paths)):
    sys.path.insert(0,paths[l])
    

isMEG = True
feat_name = ['conv1','fc7','common']
pairs = [[1,0],[0,2],[1,2]]  
n_pair = len(pairs) 
pair_names = ['Layer7-Layer1','Layer1-Common','Layer7-Common']
MEGorEEG = ['EEG','MEG']
n_comp = 6
flag_L21 = True
wsize = 16
tstep = 4


labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/" 
stc_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/%s/" %MEGorEEG[isMEG]
datadir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"




# ROIs
ROI_partition_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"
ROI_bihemi_names = ['pericalcarine', 'PPA_c_g','TOS_c_g', 'RSC_c_g','LOC_c_g']
nROI0 = len(ROI_bihemi_names)

n_part = 3
quantile_flag = False
nROI0 = len(ROI_bihemi_names)
for l in range(n_part):
    ROI_bihemi_names.append("vent_%d_qtl%d" %(l, quantile_flag))  

nROI = len(ROI_bihemi_names) 




if isMEG:
    subj_list = range(1,19)
    times = np.arange(-0.05,1.0,0.01)
else:
    subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,16,18]
    times = np.arange(-0.05,0.99,0.01)
n_subj = len(subj_list)

n_times = len(times)
Z_ts_nmlz_tssq_roi_mean = np.zeros([n_subj, nROI, 3, n_times])
Z_ts_ratio_roi_pair_mean = np.zeros([n_subj, nROI, n_pair, n_times])



for ii in range(n_subj):
    subj = "Subj%d" % subj_list[ii]
    if flag_L21:
        B = 0   
        mat_name = stc_outdir + "%s_STFT-R_all_image_Layer_1_7_CCA_ncomp%d_%s" % (subj, n_comp, MEGorEEG[isMEG])  
        mat_dict = scipy.io.loadmat(mat_name)
        active_set = mat_dict['active_set'][0]>0
       
        
        # compute the resulting time series coefficients
        n_active =active_set.sum()
        Z_all = np.zeros([B+1, n_active, mat_dict['Z'].shape[1]], dtype = np.complex)
        Z_all[0] = mat_dict['Z']
        
        
        p = 3*n_comp
        offset = 0.04 if isMEG else 0.02
        times_correct =times-offset
        
    
        # Subject 16 has an extra time point
        if subj in ["Subj16", "Subj18"] and not isMEG:
             n_step = int(np.ceil((n_times+1)/float(tstep)))
        else:
            n_step = int(np.ceil((n_times)/float(tstep)))
        n_freq = wsize// 2+1
        
        Z_ts_all = np.zeros([B+1, n_active,p,n_times])
        for l in range(B+1):
            Z_reshape_p_coef = np.reshape(Z_all[l], [n_active,p,-1])
            for i in range(p):
                tmp = Z_reshape_p_coef[:,i].reshape([n_active,n_freq, n_step])
                Z_ts_all[l,:,i] = mne.time_frequency.istft(tmp, tstep = tstep, Tx = n_times)
                
        # normalize by sum of squares of data of all time points
        Z_ts_all_three_group = np.reshape(Z_ts_all,[B+1, n_active, 3,p//3, n_times])
        # [B+1, n_active, 3, n_times]
        Z_ts_all_sumsq_three_group = np.sum(Z_ts_all_three_group**2, axis = 3)
        # Z_ts_all_sumsq_three_group shape(1, 3500 n_active, 3, 105 n_times)        
        Z_ts_nmlz_tssq = (Z_ts_all_sumsq_three_group.transpose([2,3,0,1]) \
                     /(Z_ts_all_sumsq_three_group.sum(axis = -1)).sum(axis = -1)).transpose([2,3,0,1])
        Z_ts_ratio = (Z_ts_all_sumsq_three_group.transpose([2,0,1,3])/ \
                         Z_ts_all_sumsq_three_group.sum(axis = 2)).transpose([1,2,0,3])
        # there must be NaNs
        Z_ts_ratio[np.isnan(Z_ts_ratio)] = 0.0
        # compute the pairwize difference
        Z_ts_ratio_pair = np.zeros([B+1,n_active,n_pair, n_times])
        for l0 in range(n_pair):
            Z_ts_ratio_pair[:,:,l0,:] = Z_ts_ratio[:,:, pairs[l0][0],:] - Z_ts_ratio[:,:, pairs[l0][1],:]
        
        
        stc_name0 = list()
        stc_data_to_save = list()
        for l0 in range(3):
            stc_data_to_save.append(Z_ts_nmlz_tssq[0,:,l0,:])
            stc_name0.append(feat_name[l0]+"_nmlz_tssq")
        for l0 in range(n_pair):
            stc_data_to_save.append(Z_ts_ratio_pair[0,:,l0,:])
            stc_name0.append(pair_names[l0]+"_ratio_pair")


        # save as stc, can be commented
        if False:
            if isMEG:
                fwd_path = datadir + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
            else:
                fwd_path = datadir + "EEG_DATA/DATA/fwd/%s_EEG/%s_EEG-fwd.fif" %(subj, subj)
        
            fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
            src = fwd['src']
            vertices = [src[0]['vertno'], src[1]['vertno']]
            
            for l0 in range(len(stc_data_to_save)):
                stc_data = np.zeros([len(active_set),n_times])
                stc_data[active_set,:] = stc_data_to_save[l0]
                stc = mne.SourceEstimate(data = stc_data,vertices = vertices, 
                                         tmin = times_correct[0], tstep = times_correct[2]-times_correct[1] )
                stc_name = stc_outdir + "stc/%s_%s_%s" %( subj, stc_name0[l0], MEGorEEG[isMEG])
                stc.save(stc_name) 
    
       
        #=== get results within ROIs, obtain ROI list=========================
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
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                               + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
        fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
        src = fwd['src']
        ROI_ind = list()
        for j in range(nROI0):
            tmp_label = labels_bihemi[j]
            _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
            ROI_ind.append(tmp_src_sel)
   
        if True:
            # add ROI indices
            stc_name = ROI_partition_dir + "%s/labels/%s_ventral_divide%d_label_quantile%d" %(subj, subj, n_part, quantile_flag)
            tmp_stc = mne.source_estimate.read_source_estimate(stc_name)
            dipole_label_val = tmp_stc.data[:,0]
            for i0 in range(n_part):
                tmp_ind = np.all( np.vstack([ dipole_label_val >= i0+0.5, dipole_label_val < i0+1.5]), axis = 0)
                ROI_ind.append(np.nonzero(tmp_ind)[0])
            del(stc_name, tmp_stc, dipole_label_val) 
         
        # for each ROI, average the non-zero dipoles
        active_ind_list = np.nonzero(active_set)[0]
        for j in range(nROI):
            tmp_ind = [i00 for i00 in range(len(active_ind_list)) if active_ind_list[i00] in ROI_ind[j]]
            if len(tmp_ind) == 0:
                print ROI_bihemi_names[j] + "empty"
                Z_ts_nmlz_tssq_roi_mean[ii,j,:,:] = np.NaN
                Z_ts_ratio_roi_pair_mean[ii,j,:,:] = np.NaN
            else:
                Z_ts_nmlz_tssq_roi_mean[ii,j,:,:] = (Z_ts_nmlz_tssq[0,tmp_ind,:,:]).mean(axis = 0)
                Z_ts_ratio_roi_pair_mean[ii,j,:,:] = (Z_ts_ratio_pair[0,tmp_ind,:,:]).mean(axis = 0)

"""        
plt.figure()
plt.plot(times,Z_ts_nmlz_tssq_roi_mean[ii,j].T)
plt.savefig('/home/ying/Dropbox/Scene_MEG_EEG/figs/stftr_test.png')
"""        
 
          
mat_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_regression/stftr"        
mat_name =  mat_outdir  + "%s_stftr_ROI_CCA%d.mat" %(MEGorEEG[isMEG],n_comp)

mat_dict = dict(Z_ts_nmlz_tssq_roi_mean =  Z_ts_nmlz_tssq_roi_mean,
                Z_ts_ratio_roi_pair_mean = Z_ts_ratio_roi_pair_mean,
                ROI_bihemi_names = ROI_bihemi_names,
                feat_name = feat_name, pairs = pairs, pair_names = pair_names,
                times = times_correct)
scipy.io.savemat(mat_name, mat_dict)   

          
# plotting should be shared with dSPM results
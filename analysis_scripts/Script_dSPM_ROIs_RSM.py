import numpy as np
import scipy.io

import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy
import scipy.io
import scipy.spatial
from copy import deepcopy
import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)

from RSM import get_rsm_correlation
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from Stat_Utility import bootstrap_mean_array_across_subjects

ROI_bihemi_names = [ 'pericalcarine', 
                    'PPA_c_g', 'TOS_c_g', 'RSC_c_g', 'LO_c_g',
                    'inferiortemporal', 'lateraloccipital', 'fusiform',
                    'insula', 'lateralorbitofrontal',  'medialorbitofrontal']
nROI = len(ROI_bihemi_names)                    
labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"  
MEGorEEG = ['EEG','MEG']
isMEG = True
# For now for MEG only
stc_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/" \
            + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]
fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    
    
times = np.arange(0.01,0.96,0.01)
n_times = 95
times_in_ms = times*1000.0

# load the design matrix
subj_list = [1,2,3,4,5,6,7,8,9,10,12,13]    
#subj_list = np.arange(1,14)
n_subj = len(subj_list)


feature_suffix = "no_aspect"
feat_name_seq = ["neil_attr", "neil_low", "neil_scene"]
layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]
model_names = ["AlexNet"]
for model_name in model_names:
    for layer in layers:
        feat_name_seq.append( "%s_%s_%s"%(model_name, layer, feature_suffix))

pairs = [[10,3],[8,3]]  
n_pair = len(pairs)
pair_names = [] 
for j in range(n_pair):
    pair_names.append("%s-%s" %(feat_name_seq[pairs[j][0]],feat_name_seq[pairs[j][1]]))
    

n_feat = len(feat_name_seq)
n_im = 362
mask = np.ones([n_im,n_im])
mask = np.triu(mask,1)
X_rsm_all = np.zeros([n_feat, n_im*(n_im-1)//2])
for j in range(n_feat):
    feat_name = feat_name_seq[j]
    print feat_name
    if feat_name in ["neil_attr", "neil_low", "neil_scene"]:
        # load the data X
        mat_data = scipy.io.loadmat('/home/ying/dropbox_unsync/MEG_scene_neil/PTB_Experiment/selected_image_second_round_data.mat');
        neil_attr_score = mat_data['attr_score']
        neil_low_level = mat_data['low_level_feat']
        is_high = mat_data['is_high'][:,0]
        neil_scene_score = mat_data['scene_score']
        if feat_name in [ "neil_attr"]:
            X0 = neil_attr_score 
        elif feat_name in [ "neil_low"]:
            X0 = neil_low_level 
        elif feat_name in ["neil_scene"] :
            X0 = neil_scene_score            
    elif feat_name in["rawpixel", "rawpixelgray","rawpixelgraybox"]:
        mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/Pixel_features/%s.mat" % feat_name)
        X0 = mat_data['X']
    elif feat_name in ['sun_hierarchy']:
        mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/StimSUN_semantic_feat/sun_hierarchy.mat")
        X0 = mat_data['data']  
    else:
        model_name = feat_name.split("_")[0] 
        mat_data = scipy.io.loadmat("/home/ying/Dropbox/Scene_MEG_EEG/Features/%s_Features/%s.mat" %(model_name, feat_name))
        X0 = mat_data['data']
        
        
    # for each feature, always demean them first
    X = X0- np.mean(X0, axis = 0)
    X_rsm = 1-np.corrcoef(X)
    X_rsm = X_rsm[mask > 0]
    X_rsm_all[j] = X_rsm
        

corr_ts = np.zeros([n_subj,nROI,n_feat, n_times])
for i in range(n_subj):
    subj = "Subj%d" %subj_list[i]
    labeldir1 = labeldir + "%s/" % subj
    # load and merge the labels
    labels_bihemi = list()
    for j in ROI_bihemi_names:
        tmp_label_list = list()
        for hemi in ['lh','rh']:
            print subj, j, hemi
            tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,j,hemi)
            tmp_label = mne.read_label(tmp_label_path)
            tmp_label_list.append(tmp_label)
        labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1]) 
    fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                           + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
    fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
    src = fwd['src']
    ROI_ind = list()
    for j in range(nROI):
        tmp_label = labels_bihemi[j]
        _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
        ROI_ind.append(tmp_src_sel)
        # load the source solution
        mat_dir = stc_out_dir + "%s_%s_%s_ave.mat" %(subj, MEGorEEG[isMEG],fname_suffix)
        mat_dict = scipy.io.loadmat(mat_dir)
        source_data = mat_dict['source_data']
        times = mat_dict['times'][0]
        n_times = len(times)
        del(mat_dict)
        
    ROI_data = list()
    for j in range(nROI):
        tmp = source_data[:,ROI_ind[j],:]
        tmp -= np.mean(tmp, axis = 0)
        ROI_data.append(tmp)
    del(source_data)
    
    for j in range(nROI):
        for j1 in range(n_feat):
            tmp_result = get_rsm_correlation(ROI_data[j], X, n_perm = 9,
                             perm_seq = None, metric = "correlation", demean = True, alpha = 0.05,
                             X_rsm = X_rsm_all[j1])   
            corr_ts[i,j,j1] = tmp_result['corr_ts']  

        
   
diff_corr_ts = np.zeros([n_subj,nROI,n_pair,n_times])
for l in range(n_pair):
    diff_corr_ts[:,:,l,:] = corr_ts[:,:,pairs[l][0]] - corr_ts[:,:,pairs[l][1]]
        
offset = 0.04       
time_in_ms = (times-offset) *1000    
# for each ROI, compute and plot the -log10p
fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_rsm/"

NCURVES =n_feat 
values = range(NCURVES)
jet = cm = plt.get_cmap('jet') 
import matplotlib.colors as colors
import matplotlib.cm as cmx
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

figsize = (10,10)
plt.figure(figsize = figsize)
n1,n2 = 4,3
feat_ind = np.array([3,8])
for j in range(nROI):
    ax = plt.subplot(n1,n2,j+1)   
    for l in range(len(feat_ind)):
        tmp = bootstrap_mean_array_across_subjects(corr_ts[:,j,feat_ind[l]], alpha = 0.05/n_times/3/nROI)
        tmp_mean = tmp['mean']
        tmp_se = tmp['se']
        ub = tmp['ub']
        lb = tmp['lb'] 
        colorVal = scalarMap.to_rgba(feat_ind[l])
        _ = ax.plot(time_in_ms, tmp_mean, color = colorVal)
        _ = ax.fill_between(time_in_ms, ub, lb, alpha=0.4, facecolor = colorVal) 
        _ = plt.title(ROI_bihemi_names[j])
        _ = plt.xlabel('time (ms)')
        _ = plt.ylabel('-log10(p)')
        _ = plt.grid()
    
#_=plt.legend(feat_name_seq)
#_=plt.savefig(fig_outdir + "dSPM_%dROIs_RSM.pdf" %nROI)

# diff
#plt.figure(figsize = figsize)
#count0 = 3
col2 = ['c','y','m']
for j in range(nROI):
    ax = plt.subplot(n1,n2,j+1)  
    for l in range(1,2):
        tmp = bootstrap_mean_array_across_subjects(diff_corr_ts[:,j,l,:], alpha = 0.05/n_times/3/nROI)
        tmp_mean = tmp['mean']
        tmp_se = tmp['se']
        ub = tmp['ub']
        lb = tmp['lb'] 
        _ = ax.plot(time_in_ms, tmp_mean, col2[l])
        _ = ax.fill_between(time_in_ms, ub, lb, facecolor=col2[l], alpha=0.4) 
        #_ = plt.title(ROI_bihemi_names[j] + "\n" + pair_names[l])
        _ = ax.plot(time_in_ms, np.zeros(time_in_ms.shape), 'k')
_ = plt.tight_layout(0.001)

if False:
        threshold = 1
        Tobs, clusters, p_val_clusters, H0 = mne.stats.permutation_cluster_1samp_test(
        diff_corr_ts[:,j,l,:], threshold,tail = 0)
        print clusters, p_val_clusters
        cluster_p_thresh = 0.05
        tmp_window = list()
        count = 0
        for i_c, c in enumerate(clusters):
            c = c[0]
            text_y = [1,1.2]
            if p_val_clusters[i_c] <= cluster_p_thresh:
                print count
                count = count+1
                _ = ax.axvspan(time_in_ms[c.start], time_in_ms[c.stop - 1],
                                    color='k', alpha=0.4)
                print count, l, text_y[np.mod(count,2)]     
                _ = plt.text(time_in_ms[c.start],text_y[np.mod(count,2)],('p = %1.3f' %p_val_clusters[i_c]))
                tmp_window.append(dict(start = c.start, stop = c.stop, p = p_val_clusters[i_c]))  
                
        for i in range(len(tmp_window)): 
            if tmp_window[i]['p'] <=  cluster_p_thresh:
                ax.plot(np.array([times_in_ms[tmp_window[i]['start']],  times_in_ms[tmp_window[i]['stop']]]),
                        np.array([-count0*0.3, -count0*0.3]), color =col[l], lw = 2)
                
        _ = plt.xlabel('time (ms)')
        _ = plt.ylabel('-log10(p)')
        #_ = plt.ylim(-1.5,1.5)
        _ = plt.grid()
    _ = plt.tight_layout()  
#plt.legend(['Layer 6 -1'], loc = "lower right")
#_=plt.savefig(fig_outdir + "dSPM_%dROIs_Layer6-1_PLS.pdf" %nROI)


# compare the difference of difference between ROIs?
plt.figure()
plt.imshow(diff_corr_ts[:,4,0]-diff_corr_ts[:,0,0], interpolation = "none", aspect = "auto",
           extent = [times_in_ms[0], times_in_ms[-1], 0, n_subj], origin = "lower")

tmp = bootstrap_mean_array_across_subjects(diff_corr_ts[:,4,0]-diff_corr_ts[:,0,0], alpha = 0.05/n_times/3/nROI)
tmp_mean = tmp['mean']
tmp_se = tmp['se']
ub = tmp['ub']
lb = tmp['lb'] 
_ = plt.plot(time_in_ms, tmp_mean)
_ = plt.fill_between(time_in_ms, ub, lb, alpha=0.4) 
_ = plt.grid()       
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import mne, time
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import scipy.io

import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)

from RSM import get_rsm_correlation
meg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/epoch_raw_data/"
eeg_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
MEGorEEG = ["EEG","MEG"] 

for isMEG in [0,1]:
    if isMEG == 1:
        Subj_list = range(1,19)
        n_times = 110

    if isMEG == 0:
        Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
        n_times = 109

    
    n_Subj = len(Subj_list)
    offset = 0.04   
    n_im = 362
    n_perm = 0 
    
    n_im = 362
    mask = np.ones([n_im, n_im])
    mask = np.triu(mask,1)
    
    # exclude sun_hierarchy from the analysis
    feat_name_seq = ["rawpixel","rawpixelgray", "rawpixelgraybox", "neil_attr", "neil_low", "neil_scene"]
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc6", "fc7", "prob"]
    model_names = ["AlexNet", "AlexNetgray","AlexNetcrop","hybridCNN","placeCNN"]
    for model_name in model_names:
        for layer in layers:
            feat_name_seq.append( "%s_%s"%(model_name, layer))
    
    n_feat = len(feat_name_seq)
    corr_ts = np.zeros([n_Subj, n_feat, n_times])
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
        t0 = time.time()
        
        for i in range(n_Subj):
            subj = "Subj%d" %Subj_list[i]
            if isMEG:
                ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
            else:
                ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
            
            #compute correlation
            ave_mat = scipy.io.loadmat(ave_mat_path)
            ave_data = ave_mat['ave_data']
            #compute correlation
            result_rdm = get_rsm_correlation(ave_data, X, n_perm = n_perm,
                             perm_seq = None, metric = "correlation", demean = True, alpha = 0.05,
                             X_rsm = X_rsm)   
            #debug:
            #result_rdm1 = get_rsm_correlation(ave_data, X, n_perm = n_perm,
            #                perm_seq = None, metric = "correlation", demean = True, alpha = 0.05,
            #                X_rsm = None)
            #print np.linalg.norm(result_rdm['corr_ts']-result_rdm1['corr_ts'])/np.linalg.norm(result_rdm1['corr_ts'])
            result_rdm['times'] = ave_mat['times'][0][0:n_times]
            corr_ts[i,j] =result_rdm['corr_ts'][0:n_times]
                    
        print time.time() - t0
    
    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_dependence/"
    matname = outdir + "Subj_pooled_rdm_noperm_%s.mat" %(MEGorEEG[isMEG])
    matdict = dict(corr_ts = corr_ts, times = result_rdm['times'], feat_name_seq = feat_name_seq, 
                   Subj_list = Subj_list)
    scipy.io.savemat( matname, matdict)
    
 
# ==============================================================================
# load all feat name for each subject
 
if False:
    import scipy.io
    import numpy as np
    import matplotlib.pyplot as plt
    
    isMEG = 0
    MEGorEEG = ['EEG','MEG']
    

    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_dependence/"
    matname = outdir + "Subj_pooled_rdm_noperm_%s.mat" %(MEGorEEG[isMEG])
    matdict = scipy.io.loadmat(matname) 
    
    corr_ts, times = matdict['corr_ts'], matdict['times'][0]
    feat_name_seq = matdict['feat_name_seq']
     
    n_feat_name = len(feat_name_seq)
    n_Subj = corr_ts.shape[0]
    Subj_list = matdict['Subj_list'][0]
    n_times = len(times)
    offset = 0.04 if isMEG else 0.00
    times_in_ms = (times - offset)*1000.0
    plt.figure(figsize = (12,12))
    ymin, ymax  = -0.015, 0.20
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_rsm/"
    
    if False:
        # compare between all AlexNet ones
        corr_ts1 = corr_ts[:,5:8*3+5,:]
        corr_ts1 = corr_ts1.reshape([corr_ts1.shape[0], 3,8, -1])
        feat_name_seq1 = (feat_name_seq[5:8*3+5]).reshape([3,8])
        # plot the pairwise difference T-stat
        pair_seq = [[0,1],[0,2]]
        plt.figure()
        count = 1
        vmin, vmax = None,None
        for l in range(len(pair_seq)):
            for l1 in range(8):
                _=plt.subplot(len(pair_seq),8,count)
                _=tmp_diff = corr_ts1[:,pair_seq[l][0],l1,:]-corr_ts1[:,pair_seq[l][1],l1,:]
                _=plt.imshow(tmp_diff, interpolation = "none", aspect = "auto", vmin=vmin, vmax = vmax)
                _=plt.colorbar()
                _=plt.title("%s\n%s" %(feat_name_seq1[pair_seq[l][0],l1], feat_name_seq1[pair_seq[l][1],l1]))
                count += 1
        #_= plt.tight_layout()
                
    if True:          
        tmp_ind = np.arange(6,5+8*3,1)
        corr_ts2 =  corr_ts[:,tmp_ind,:] 
        feat_name_seq2 = feat_name_seq[tmp_ind]
        plt.figure()
        m1, m2 = 4,5
        for i in range(n_Subj):
            ax = plt.subplot(m1,m2,i+1)
            #NCURVES = len(feat_name_seq)
            NCURVES =len(tmp_ind)
            #NCURVES = 3
            values = range(NCURVES)
            jet = cm = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            #print scalarMap.get_clim() 
            lines = []
            for idx in range(NCURVES):
                
                line = corr_ts2[i,idx]
                colorVal = scalarMap.to_rgba(values[idx])
                colorText = (
                    'color: (%4.2f,%4.2f,%4.2f)'%(colorVal[0],colorVal[1],colorVal[2])
                    )
                retLine, = ax.plot(times_in_ms, line,
                                   color=colorVal)
            _=plt.ylim( ymin, ymax)
            _=plt.title("Subj%d" % Subj_list[i])
            _=plt.xlabel('time (ms)')
          
        #_=plt.legend(feat_name_seq2[0:NCURVES], loc = 'upper right', fontsize = '5')
        _=plt.tight_layout()
        _=plt.savefig(fig_outdir + "Subj_pooled_all_feat_%s.pdf" %MEGorEEG[isMEG])
    
    
    plt.figure(figsize =(15,10))
    mean_corr_ts = np.mean(corr_ts, axis = 0)
    _=plt.imshow(mean_corr_ts, aspect = "auto", interpolation = "None")
    _=plt.yticks(np.arange(n_feat_name)+0.5, feat_name_seq)
    _=x_ticks = np.arange(0,n_times,5)
    _=plt.xticks(x_ticks, (times_in_ms[x_ticks]).astype(np.int))
    _=plt.colorbar()
    _=plt.savefig(fig_outdir + "Mean_corrts_all_feat_%s.pdf" %MEGorEEG[isMEG])
    
    #===========================================================================
    if False:
        # plot by features
        plt.figure(figsize = (15,10))
        ymin, ymax  = -0.015, 0.15
        #ymin,ymax = None, None
        m1,m2 = 7,7
        for i in range(n_feat_name):
            ax = plt.subplot(m1,m2,i+1)
            plt.imshow(corr_ts[:,i,:], aspect = "auto", interpolation = "none",
                           vmin = ymin, vmax = ymax,
                           extent = [times_in_ms[0], times_in_ms[-1], 1, n_Subj], origin = "lower",)
        
            plt.title( feat_name_seq[i])
            plt.xlabel('time (ms)')
            plt.colorbar()
        plt.tight_layout()
        plt.savefig(fig_outdir + "Subj_pooled_by_feat_%s.pdf" %MEGorEEG[isMEG])
        
        
        #========= plot pairwise difference =================
        plt.figure(figsize = (50,50))
        count = 0
        vabs = 0.05
        n_feat_name_tmp = 11
        for i in range(n_feat_name_tmp):
            for j in range(i+1, n_feat_name_tmp):
                plt.subplot(n_feat_name_tmp-1, n_feat_name_tmp-1, i*(n_feat_name_tmp-1) + j)
                plt.imshow(corr_ts[:,i,:]- corr_ts[:,j,:], aspect = "auto", interpolation = "none",
                           vmin = -vabs, vmax = vabs,
                           extent = [times_in_ms[0], times_in_ms[-1], 1, n_Subj], origin = "lower",)
                
                plt.title( "%s\n %s" % (feat_name_seq[i],feat_name_seq[j]))
                
                if j == i+1:
                    plt.xlabel('times (ms)')
                    plt.ylabel('subj id')
                else:
                    plt.axis('off') 
        
        plt.subplot(n_feat_name_tmp-1, n_feat_name_tmp-1, (n_feat_name_tmp-2)*(n_feat_name_tmp-1)+1 )
        plt.colorbar()
        plt.tight_layout(0.01)
        plt.savefig(fig_outdir + "Subj_pooled_all_feat_diff_%s.pdf" %MEGorEEG[isMEG]) 
    
        # plot the difference between the models
        feat_subset = range(4,n_feat_name)
        feat_name_subset = np.reshape(feat_name_seq[feat_subset],[3,8])
        corr_ts_subset = corr_ts[:, feat_subset, :]
        corr_ts_subset = np.reshape(corr_ts_subset, [n_Subj, 3, 8, -1])
        
        # for each layer, plot the difference between them
        
        plt.figure(figsize = (20,15))
        for i in range(8):
            plt.subplot(4,2,i+1)
            legend = list()
            for j1 in range(3):
                for j2 in range(j1+1,3):
                    tmp_diff = corr_ts_subset[:,j1,i,:]-corr_ts_subset[:,j2,i,:]
                    tmp_T = np.mean(tmp_diff,axis = 0)/np.std(tmp_diff, axis = 0)*np.sqrt(n_Subj)
                    plt.plot(times_in_ms, tmp_T)
                    legend.append("%s\n-%s" %(feat_name_subset[j1,i], feat_name_subset[j2,i]))
            plt.legend(legend)
        plt.savefig(fig_outdir + "T_diff_NN_%s.pdf" %MEGorEEG[isMEG])
        
        plt.figure(figsize = (20,15))
        for i in range(8):
            plt.subplot(4,2,i+1)
            legend = list()
            for j1 in range(3):
                for j2 in range(j1+1,3):
                    tmp_diff = corr_ts_subset[:,j1,i,:]-corr_ts_subset[:,j2,i,:]
                    plt.plot(times_in_ms,  np.mean(tmp_diff,axis = 0))
                    legend.append("%s\n-%s" %(feat_name_subset[j1,i], feat_name_subset[j2,i]))
            plt.legend(legend)
        plt.savefig(fig_outdir + "mean_diff_NN_%s.pdf" %MEGorEEG[isMEG])
        
        plt.figure(figsize = (20,15))
        count = 0
        vabs = 0.05
        for i in range(8):
            legend = list()
            for j1 in range(3):
                for j2 in range(j1+1,3):
                    count += 1
                    plt.subplot(8,3, count)
                    tmp_diff = corr_ts_subset[:,j1,i,:]-corr_ts_subset[:,j2,i,:]
                    plt.imshow(tmp_diff,  aspect = "auto", interpolation = "none",
                           vmin = -vabs, vmax = vabs,
                           extent = [times_in_ms[0], times_in_ms[-1], 1, n_Subj], origin = "lower")
                    plt.title("%s\n-%s" %(feat_name_subset[j1,i], feat_name_subset[j2,i]))
                    plt.colorbar()
        plt.tight_layout(0.01)
        plt.savefig(fig_outdir + "diff_NN_%s.pdf" %MEGorEEG[isMEG])
            
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
import scipy.io
import mne
mne.set_log_level('WARNING')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import scipy.spatial
import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from Stat_Utility import bootstrap_mean_array_across_subjects


MEGorEEG = ['EEG','MEG']
#method = "dSPM"
method = "stftr"
# load the mat files

#suffix = "_repeat"
#ROI_bihemi_names = ['EVC','Vent1','Vent2','Vent3']

suffix = ""
ROI_bihemi_names1 = ['EVC','PPA','TOS','RSC','LOC','Vent1','Vent2','Vent3']


n_comp = 6
Flag_CCA = True
isMEG = True

feat_name_seq1 = ['res Layer 1', 'res Layer 7', 'common']


for method in ['dSPM','stftr']:
    for isMEG in [False, True]:

        if method == "dSPM":
            mat_outdir ="/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_regression/dSPM_reg_ROI/"        
            mat_name = mat_outdir  + "%s_dSPM_ROI_ols_reg_CCA%d_%d.mat" %(MEGorEEG[isMEG], Flag_CCA, n_comp)
            mat_dict = scipy.io.loadmat(mat_name)
            
            val = mat_dict['val']
            #diff_val = mat_dict['diff_val']
            ROI_bihemi_names = mat_dict['ROI_bihemi_names']
            for ll in range(len(ROI_bihemi_names)):
                ROI_bihemi_names[ll] = ROI_bihemi_names[ll].split()[0]
            del(ll)
            feat_name_seq = mat_dict['feat_name_seq']
            pairs = mat_dict['pairs']
            pair_names = mat_dict['pair_names']
            n_pair = len(pairs)
            
            n_times = 100 if isMEG else 99
            times = mat_dict['times'][0][0:n_times]
         
            offset = 0.04 if isMEG else 0.02      
            time_in_ms = (times-offset) *1000 
            
            n_subj = val.shape[0]
            nROI = val.shape[1]
            n_times = len(time_in_ms)
            
            flag_mean_ratio = True
            if flag_mean_ratio:
                print "normalize before average"
                # normalized first before average
                val_normalized = mat_dict['val_normalized']
                val_normalized_timewise = mat_dict['val_normalized_timewise']
            else:  
                # val was averaged across each individual ROI first
                val_normalized = np.zeros(val.shape)
                for i in range(n_subj): 
                    for j in range(nROI): 
                        val_normalized[i,j,:,:] = val[i,j,:,:]/np.sum(val[i,j,:,:])
                
                # time-wize normalized val: ratio of the 3 features 
                val_normalized_timewise = np.zeros(val.shape)
                for i in range(n_subj): 
                    for j in range(nROI): 
                        val_normalized_timewise[i,j,:,:] = val[i,j,:,:]/np.sum(val[i,j,:,:], axis = 0)
            
            val_normalized /= np.max(val_normalized)    
            data_list = [val, val_normalized, val_normalized_timewise]
            data_list_name = ['val','val_nmlz', 'val_nmlz_twise']
            
        
            # only compare the normalized
            if n_pair > 0:
                k0 = 2
                tmp = np.zeros([n_subj, nROI, n_pair, n_times])
                for l0 in range(n_pair):
                    tmp[:,:,l0,:] = data_list[k0][:,:,pairs[l0][0],:] -data_list[k0][:,:,pairs[l0][1],:] 
                data_pair_list = [tmp]
                data_pair_list_name = ['pair_val_nmlz_twise']
                
            
            #  plot range for data
            ymin_list0 = np.zeros([3])
            ymax_list0 = np.array([7.5, 1.0, 1.0])
            # plot range for difference
            ymin_list1 =  np.array([[-1.0, -1.0, -1.0]])
            ymax_list1 =  np.array([[1.0, 1.0, 1.0]])
            
            ymin_list0 = np.zeros([3])
            ymax_list0 = np.array([0.3, 1.0, 1.0])
            # plot range for difference
            ymin_list1 =  np.array([[-0.5, -0.5, -0.5]])
            ymax_list1 =  np.array([[0.5, 0.5, 0.5]])
        
        
            
        elif method == "stftr":
            # TO BE CHANGED
            n_comp = 6
            mat_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_regression/stftr"        
            mat_name =  mat_outdir  + "%s_stftr_ROI_CCA%d.mat" %(MEGorEEG[isMEG],n_comp)
            mat_dict = scipy.io.loadmat(mat_name)
            Z_ts_nmlz_tssq_roi_mean = mat_dict['Z_ts_nmlz_tssq_roi_mean']
            Z_ts_ratio_roi_pair_mean = mat_dict['Z_ts_ratio_roi_pair_mean']
            times = mat_dict['times'][0]
             
            # time offset was already corrected
            #offset = 0.04 if isMEG else 0.02
            #times = times-offset
            time_in_ms = (times) *1000
            
            
            feat_name_seq = mat_dict['feat_name']
            pair_names = mat_dict['pair_names']
            ROI_bihemi_names = mat_dict['ROI_bihemi_names']
            
            n_subj,nROI, _, n_times = Z_ts_nmlz_tssq_roi_mean.shape
            
            pair_names = ['ResLayer7-1', 'ResLayer1-Common','ResLayer7-Common']
            data_list = [Z_ts_nmlz_tssq_roi_mean/np.nanmax(Z_ts_nmlz_tssq_roi_mean)]
            data_list_name = ['Z_ts_nmlz_tssq']
            data_pair_list = [Z_ts_ratio_roi_pair_mean]
            data_pair_list_name = ['pair_Z_ts_ratio_pair']
            
            ymin_list0 = np.zeros([1])
            ymax_list0 = np.array([1])
            # plot range for difference
            ymin_list1 =  np.array([[-0.5, -0.5, -0.5]])
            ymax_list1 =  np.array([[0.5, 0.5, 0.5]])
        
        
        #===============================================================================
        
            
        fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_reg/"
        figsize = 3.6,2.5
        #plt.figure(figsize = figsize)
        #n1 = np.int(np.ceil(np.sqrt(nROI)))
        #n2 = nROI//n1+1
        
        
        
        if not Flag_CCA:
            
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            from mpl_toolkits.axes_grid1 import make_axes_locatable
        
            NCURVES = len(feat_name_seq)
            values = range(NCURVES)
            jet = cm = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            colorVal_list = list()
            for l in range(len(feat_name_seq)):
                colorVal = scalarMap.to_rgba(values[l])
                colorVal_list.append(colorVal)
                
            
            feat_name_seq0 = np.arange(1,8)
            btstrp_alpha =  0.05/n_times/len(feat_name_seq)
            for k in range(len(data_list)):
                for j in range(nROI):
                    #ax = plt.subplot(n1,n2,j+1) 
                    plt.figure(figsize = figsize)
                    ax = plt.subplot(1,1,1)
                    for l in range(len(feat_name_seq)):
                        colorVal = colorVal_list[l]
                        # remove the NaNs
                        tmp_data = data_list[k][:,j,l]
                        valid_ind =np.all( True-np.isnan(tmp_data), axis = 1)
                        tmp_data = tmp_data[valid_ind,:]
                        
                        tmp = bootstrap_mean_array_across_subjects(tmp_data, alpha = btstrp_alpha)
                        tmp_mean = tmp['mean']
                        tmp_se = tmp['se']
                        ub = tmp['ub']
                        lb = tmp['lb'] 
                        _ = ax.plot(time_in_ms, tmp_mean, color = colorVal)
                        _ = ax.fill_between(time_in_ms, ub, lb, facecolor= colorVal, alpha=0.2) 
                        _ = plt.title(ROI_bihemi_names1[j])
                        _ = plt.xlabel('time (ms)')
                        #_ = plt.ylabel('-log10(p)')
                        _ = plt.ylim(ymin_list0[k], ymax_list0[k])
                        _ = plt.tight_layout(0.001)
                    if j == 0:
                        _=plt.legend(feat_name_seq0, fontsize = 5)
                    _=plt.savefig(fig_outdir + "%s_%s_%s_CCA%d_ncomp%d_%s%s.png"\
                            %(MEGorEEG[isMEG], method, ROI_bihemi_names1[j], Flag_CCA, n_comp, data_list_name[k],suffix), dpi = 1000.0)
                    _=plt.close('all')    
        
        else:    
            #===================== plot the scores for 3 feature sets======================
            btstrp_alpha =  0.05/n_times/len(feat_name_seq)
            col = ['b','g','r']
            for k in range(len(data_list)):
                for j in range(nROI):
                    #ax = plt.subplot(n1,n2,j+1) 
                    plt.figure(figsize = figsize)
                    ax = plt.subplot(1,1,1)
                    for l in range(len(feat_name_seq)):
                        # remove the NaNs
                        tmp_data = data_list[k][:,j,l]
                        valid_ind =np.all( True-np.isnan(tmp_data), axis = 1)
                        if np.sum(valid_ind) == 0:
                            continue
                        tmp_data = tmp_data[valid_ind,:]
                        tmp = bootstrap_mean_array_across_subjects(tmp_data, alpha = btstrp_alpha)
                        tmp_mean = tmp['mean']
                        tmp_se = tmp['se']
                        ub = tmp['ub']
                        lb = tmp['lb'] 
                        _ = ax.plot(time_in_ms, tmp_mean, col[l])
                        _ = ax.fill_between(time_in_ms, ub, lb, facecolor=col[l], alpha=0.4) 
                        _ = plt.title(ROI_bihemi_names1[j])
                        _ = plt.xlabel('time (ms)')
                        #_ = plt.ylabel('-log10(p)')
                        _ = plt.ylim(ymin_list0[k], ymax_list0[k])
                        _ = plt.tight_layout(0.001)
                    if j == 0:
                        _=plt.legend(feat_name_seq1)
                    tmp_fig_name = fig_outdir + "%s_%s_%s_CCA%d_ncomp%d_%s%s"\
                            %(MEGorEEG[isMEG], method, ROI_bihemi_names1[j], Flag_CCA, n_comp, data_list_name[k],suffix)
                    _=plt.savefig(tmp_fig_name + ".png", dpi = 1000.0)
                    _=plt.savefig(tmp_fig_name + ".pdf", dpi = 1000.0)
                    _=plt.close('all')
        
        
        #====================== plot the difference====================================
        
        pair_ind_seq = [0,1,2]
        col2 = ['c','y','m']
        # try the threshold free version TFCE
        #threshold = dict(start= 1.8, step=0.1)
        alpha0 = 0.05
        threshold = scipy.stats.t.ppf(1-alpha0/2, n_subj-1)
        
        # hard coded here 
        out_file = fig_outdir+"%s_%s_pairwise_compare_pthresh.txt" %(MEGorEEG[isMEG], method)
        if os.path.isfile(out_file):
            with open(out_file, 'r') as fid:
                cluster_p_thresh = float(fid.read())
                print cluster_p_thresh
        else:
            cluster_p_thresh = 0.05 #/len(pair_ind_seq)
            
        btstrp_alpha = 0.05/n_times/len(pair_ind_seq)
        p_all = np.zeros(0)
        
        for k in range(len(data_pair_list)):
            for j in range(nROI): 
                for l in pair_ind_seq:
                    plt.figure(figsize = figsize)
                    ax = plt.subplot(1,1,1) 
                    
                    tmp_data = data_pair_list[k][:,j,l,:]
                    valid_ind =np.all( True-np.isnan(tmp_data), axis = 1)
                    if valid_ind.sum() == 0:
                        continue
                    tmp_data = tmp_data[valid_ind,:]
                    
                    tmp = bootstrap_mean_array_across_subjects(tmp_data, alpha = btstrp_alpha)
                    tmp_mean = tmp['mean']
                    tmp_se = tmp['se']
                    ub = tmp['ub']
                    lb = tmp['lb'] 
                    _ = ax.plot(time_in_ms, tmp_mean, col2[l])
                    _ = ax.fill_between(time_in_ms, ub, lb, facecolor=col2[l], alpha=0.4) 
                    _ = plt.title(ROI_bihemi_names1[j])
                    _ = ax.plot(time_in_ms, np.zeros(time_in_ms.shape), 'k')
                    _ = plt.xlabel('time (ms)')
                    
                    #_ = plt.tight_layout(0.001)    
                    Tobs, clusters, p_val_clusters, H0 = mne.stats.permutation_cluster_1samp_test(tmp_data, threshold,tail = 0)
                    print clusters, p_val_clusters
                    tmp_window = list()
                    count0 = 4
                    count = 0
                    for i_c, c in enumerate(clusters):
                        if not isinstance(threshold, dict): 
                            c = c[0]
                        text_y = np.array([0.3,-0.3, 0.5,-0.5])*ymax_list1[k,l]
                        if p_val_clusters[i_c] <= cluster_p_thresh:
                            p_all = np.hstack([p_all, p_val_clusters[i_c]])
                            print count
                            count = count+1
                            _ = ax.axvspan(time_in_ms[c.start], time_in_ms[c.stop - 1],
                                                color='k', alpha=0.1)
                            print count, l, text_y[np.mod(count,4)]     
                            _ = plt.text(time_in_ms[c.start],text_y[np.mod(count,4)],('p = %1.3f' %(p_val_clusters[i_c])))
                            tmp_window.append(dict(start = c.start, stop = c.stop, p = p_val_clusters[i_c]))
                    if False:
                        for i in range(len(tmp_window)): 
                            if tmp_window[i]['p'] <=  cluster_p_thresh:
                                ax.plot(np.array([time_in_ms[tmp_window[i]['start']],  time_in_ms[tmp_window[i]['stop']-1]]),
                                        np.array([-count0*0.3, -count0*0.3]), color =col2[l], lw = 2)
                            
                    _ = plt.ylim(ymin_list1[k,l], ymax_list1[k,l])
                    tmp_fig_name = fig_outdir + "%s_%s_%s_CCA%d_ncomp%d_diff_%s_%s%s"\
                            %(MEGorEEG[isMEG], method, ROI_bihemi_names1[j], Flag_CCA, n_comp, pair_names[l], data_list_name[k],suffix)
                    _= plt.tight_layout(0.001)
                    _=plt.savefig(tmp_fig_name + ".png", dpi = 1000.0)
                    _=plt.savefig(tmp_fig_name + ".pdf", dpi = 1000.0)
                    
        if len(p_all) > 0:                   
            #fdr_true, _ = mne.stats.fdr_correction(p_all, method = "negcorr") 
            print "p_all"
            print p_all
            fdr_true, _ = mne.stats.fdr_correction(p_all, method = "indep") 
            out_file = fig_outdir+"%s_%s_pairwise_compare_pthresh.txt" %(MEGorEEG[isMEG], method)
            
            print p_all[fdr_true].max()
            with open(out_file, 'w') as fid:
                fid.write( "%1.5f" % p_all[fdr_true].max())
                fid.close()
                                
        #========== hand coded
        # compute the difference of the difference normalized
        '''
                          
        k = 0
        l = 2
        diff_fc6_common = data_pair_list[k][:,:,l,:]
        
        threshold = 3.0
        cluster_p_thresh = 0.01
        # part 2-part 1
        roi_pairs = [[5,0],[6,5],[7,6]]
        for ll in range(len(roi_pairs)):
            roi_id1 = roi_pairs[ll][0]
            roi_id2 = roi_pairs[ll][1]
            plt.figure(figsize = figsize)
            ax = plt.subplot(1,1,1) 
            tmp_data= diff_fc6_common[:, roi_id1] - diff_fc6_common[:, roi_id2]
           
            valid_ind =np.all( True-np.isnan(tmp_data), axis = 1)
            tmp_data = tmp_data[valid_ind,:]
                    
            tmp = bootstrap_mean_array_across_subjects(tmp_data, alpha = 0.05/n_times)
            tmp_mean = tmp['mean']
            tmp_se = tmp['se']
            ub = tmp['ub']
            lb = tmp['lb'] 
            _ = ax.plot(time_in_ms, tmp_mean, 'm')
            _ = ax.fill_between(time_in_ms, ub, lb, facecolor='m', alpha=0.4) 
            _ = plt.title(ROI_bihemi_names[roi_id1] +"-"+ ROI_bihemi_names[roi_id2])
            _ = ax.plot(time_in_ms, np.zeros(time_in_ms.shape), 'k')
            
            #_ = plt.tight_layout(0.001)    
            Tobs, clusters, p_val_clusters, H0 = mne.stats.permutation_cluster_1samp_test(tmp_data, threshold,tail = 0)
            print clusters, p_val_clusters
            tmp_window = list()
            count0 = 3
            count = 0
            for i_c, c in enumerate(clusters):
                c = c[0]
                text_y = np.array([0.7,-0.7,-0.9])*ymax_list1[k,l]
                if p_val_clusters[i_c] <= cluster_p_thresh:
                    print count
                    count = count+1
                    _ = ax.axvspan(time_in_ms[c.start], time_in_ms[c.stop - 1],
                                        color='k', alpha=0.1)
                    print count, l, text_y[np.mod(count,3)]     
                    _ = plt.text(time_in_ms[c.start],text_y[np.mod(count,3)],('p = %1.3f' %(p_val_clusters[i_c])))
                    tmp_window.append(dict(start = c.start, stop = c.stop, p = p_val_clusters[i_c]))  
            if False:
                for i in range(len(tmp_window)): 
                    if tmp_window[i]['p'] <=  cluster_p_thresh:
                        ax.plot(np.array([time_in_ms[tmp_window[i]['start']],  time_in_ms[tmp_window[i]['stop']-1]]),
                                np.array([-count0*0.3, -count0*0.3]), color =col2[l], lw = 2)
                    
            _ = plt.ylim(ymin_list1[k,l], ymax_list1[k,l])
            _ = plt.savefig(fig_outdir + "%s_Layer1_6_CCA%d_diff_%s_%s_ROI_%s_%s%s.png"\
                    %(method, n_comp, pair_names[l], data_list_name[k], ROI_bihemi_names[roi_id1], ROI_bihemi_names[roi_id2],suffix), dpi = 1000.0)
        
        '''    

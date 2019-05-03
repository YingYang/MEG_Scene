# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import time
import matplotlib.colors as colors
import matplotlib.cm as cmx
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
#path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/"

sys.path.insert(0, path0)
#path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
path1 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
#
sys.path.insert(0, path1)

path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from Stat_Utility import bootstrap_mean_array_across_subjects

import time

from matplotlib import ticker
# sensor ave significance function
 #edgecolors = 'k'
 
 


#===============================
def visualize_sensors( tmp, vmin, vmax, figname, cmap, figsize, isMEG,
                      times, time_array, Flag_scatter = True):
    
    if isMEG:
        data = scipy.io.loadmat("/media/yy/LinuxData/yy_dropbox/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/" \
                                      + "Utility/sensor_layout.mat")
        pos = data['position'][:,0:2]                              
    else:
        # to be updated
        layoutpath = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/biosemi128_layout/"
        layout = mne.channels.read_layout("biosemi128", layoutpath) 
        pos = layout.pos[:,0:2]


    n = len(time_array)
    half_window = 3
    tick_locator = ticker.MaxNLocator(nbins=5)

    eps = 5E-4
    fig = plt.figure(figsize = figsize)
    
    
    lw = 0.05
    edgecolors = "none"  
    area = 12
    
    for i in range(n):
        ax = fig.add_subplot(1,n,i+1)
        time_ind = np.where( np.abs(times -time_array[i]) < eps)[0][0]
        z = np.mean(tmp[:,(time_ind-half_window):(time_ind+half_window)], axis = 1)
        
        if Flag_scatter:
            _ = ax.scatter(pos[:,0], pos[:,1], c = z, s = area, lw = lw,
                       vmin = vmin, vmax = vmax, cmap = cmap,
                       edgecolors= edgecolors) 
            _ = plt.xlim(-0.01,1.01)
            _ = plt.ylim(-0.01,1.01) 
            #plt.gca().set_aspect('equal', adjustable='box')               
            _ = plt.axis('off') 
            #_ = plt.plot(pos[:,0],pos[:,1],'.k', markersize = 1) 
        else:
            _ = mne.viz.plot_topomap(z, pos[:,0:2], vmin=vmin, vmax=vmax, cmap=cmap,
                 axes=ax, contours = False) 
            # can not do title?
            # or I can create an evoked object, and do plot
        
        _ = ax.set_title( "%dms " % (time_array[i]*1000) ) 
 
    fig.subplots_adjust(hspace = 0.001, wspace = 0.001, right = 0.9) 
    cbar_ax = fig.add_axes([0.9, 0.25, 0.01, 0.6])
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # ColorbarBase derives from ScalarMappable and puts a colorbar
    # in a specified axes, so it has everything needed for a
    # standalone colorbar.  There are many more kwargs, but the
    # following gives a basic continuous colorbar with ticks
    # and labels.
    cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=cmap,norm=norm)
    cb.locator = tick_locator
    cb.update_ticks()   
    plt.savefig(figname, bbox_inches='tight')





def visualize_sensors_MEG_topo( info1, value, vmin, vmax, figname, figsize, 
                      times, time_array, title = " ", ylim = [-2,10]):
    
    '''
    value must be [n_channel, n_times]
    '''

    value1 = np.reshape(value, [102,3, value.shape[1]]).mean(axis = 1)
    value2 = value.copy()
    for i in range(3):
       value2[i::3] = value1
       
    value2 = value.copy()

    evoked = mne.EvokedArray(value2, info1, tmin = times[0], nave = None)
    joint_kwargs = dict(ts_args=dict(time_unit='ms',  units = dict(grad = "% variance", 
                                    mag = "% variance", eeg = "% variance"), scalings = dict(grad = 1, mag = 1, eeg = 1),
                                    ylim = dict(eeg = ylim, grad = ylim, mag = ylim),
                                    ),
                        topomap_args=dict(time_unit='ms', units = "%", vmin = vmin, vmax = vmax,
                                          scalings = dict(grad = 1, mag = 1, eeg = 1),
                                          contours = 0))
    fig, fig2 = evoked.plot_joint(times=time_array, title = title, **joint_kwargs)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(figsize)
    fig.savefig(figname[0:-4] +".pdf", bbox_inches='tight')
    fig.savefig(figname[0:-4] +".png", bbox_inches='tight', dpi = 300)

# plain imshow functions
# TBA
#=======================================================================================
if __name__ == "__main__":
    
    MEG_DATA_DIR = "/Users/yingyang/Downloads/Scene_MEG_data/"
    REGRESSOR_DIR = "/Users/yingyang/Downloads/essential_regressor/"
    MAT_OUT_DIR = "/Users/yingyang/Downloads/result/"
        
    FIG_OUT_DIR = "/Users/yingyang/Downloads/result/"


    
    # try unsmoothed
    #fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    
    flag_swap_PPO10_POO10 = True
    MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    
    model_name = "AlexNet"


    #fname = "/media/yy/LinuxData/yy_Scene_MEG_data/MEG_preprocessed_data/MEG_preprocessed_data/epoch_raw_data/Subj1/Subj1_run1_filter_1_110Hz_notch_ica_smoothed-epo.fif.gz"
    fname = "/Users/yingyang/mne_data/MNE-sample-data/MEG/sample/sample_audvis-ave.fif"
    tmp_epoch = mne.read_evokeds(fname)
    info1 = tmp_epoch[0].info
    del(tmp_epoch)
    
    info1['ch_names'] = info1['ch_names'][0:306]
    info1['chs'] = info1['chs'][0:306]
    info1['bads'] = []
    info1['nchan'] = 306
    info1['sfreq'] = 100.0
    

    
    Flag_CCA = True
    Flag_PCA = True
    
    Flag_additional = False
    Flag_cv = False
    #"AlexNet_MEG_CCA1_ncomp6_ave_cv_1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    
    #for isMEG in [0,1]:
    isMEG = True
    if True:
        if isMEG:
            Subj_list = range(1,19)
            n_times = 110
            n_channels = 306
        else:
            Subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
            n_times = 109
            n_channels = 128
                
        n_Subj = len(Subj_list)
        n_im = 362
            
        offset = 0.04 if isMEG else 0.02
        # if not Flag_PCA, the ridge regression
        #========= the first 10 dimension
             
        if Flag_PCA:
           
            if not Flag_CCA:
                feature_suffix = "no_aspect_no_contrast160_all_im"
                #feature_suffix = "no_aspect"
                #feature_suffix = "no_aspect_no_contrast100"
                n_comp = 10
            else:
                n_comp = 6
                feature_suffix = "CCA" 
            
            fname_suffix = MEG_fname_suffix 
            if Flag_cv:
                mat_name = MAT_OUT_DIR + "AlexNet_%s_%s_ncomp%d_ave_cv_%s.mat" \
                    %("MEG", feature_suffix, n_comp, fname_suffix)
                mat_dict = scipy.io.loadmat(mat_name)
                Rsq, times = 1.0-mat_dict['relative_error'], mat_dict['times'][0]
            else:
                mat_name = MAT_OUT_DIR + "AlexNet_%s_%s_ncomp%d_ave_%s.mat" \
                    %("MEG", feature_suffix, n_comp, fname_suffix)
                mat_dict = scipy.io.loadmat(mat_name)
                Rsq, times = mat_dict['Rsq'], mat_dict['times'][0]
            
           
            feat_name_seq = mat_dict['feat_name_seq']
            feat_name1 = mat_dict['feat_name1']
            n_feat = len(feat_name_seq)
            
            for i in range(n_feat):
                feat_name_seq[i] = feat_name_seq[i].split()[0]
            
            times1 = (times-offset)
            times_ms =times1*1000.0
            vmin = -0.05 if Flag_cv else  0
            vmax_list = [0.08] if Flag_cv else [0.08]
            time_array = np.arange(0.05,0.75,0.05)
            

                
    
        data_to_plot = [Rsq]
        data_name = ['Rsq'] 
        """
        if Flag_CCA:
            n1, n2 = 1, n_feat
            figsize = (14,5)
        else:
            n1, n2 = 3, np.ceil(n_feat/3.0)
            figsize = (14,14)
        for ll in range(len(data_to_plot)):    
            plt.figure(figsize = figsize)
            for j in range(n_feat):
                tmp = np.mean(data_to_plot[ll][:,j,:,:], axis = 0)
                tmp_max = np.max(tmp)
                _= plt.subplot(n1,n2,j+1); 
                _= plt.imshow(tmp,interpolation = "none", vmin = vmin, vmax = vmax_list[ll],
                            extent = [times_ms[0], times_ms[-1], 0, n_channels],
                            origin = "lower", aspect = "auto")
                _= plt.title(feat_name_seq[j] + " max = %1.2f" %tmp_max); 
                _= plt.colorbar();
                if j == 0:
                    _= plt.xlabel('time (ms)');
                    _= plt.ylabel('channel id');
        visualize_sensors_MEG_topo    plt.tight_layout();
                if Flag_CCA:
                    plt.savefig(fig_outdir + "AlexNet_%s_CCA%d_ncomp%d_ave_%s.eps" %(MEGorEEG[isMEG], sep_CCA, n_comp, data_name[ll]))
                else:
                    plt.savefig(fig_outdir + "AlexNet_%s_%s_ncomp%d_ave_%s.eps" %(MEGorEEG[isMEG], feature_suffix, n_comp, data_name[ll]))
                    
                # difference
                if Flag_CCA:
                    plt.figure(figsize = (14,5))
                    count = 0
                    for i in range(n_feat):
                        for j in range(i+1,n_feat):
                            count += 1
                            tmp_diff = data_to_plot[ll][:,j,:,:] - data_to_plot[ll][:,i,:,:]
                            tmp_T_diff = np.mean(tmp_diff, axis = 0)/np.std(tmp_diff, axis = 0)*np.sqrt(n_Subj)
                            plt.subplot(1,n_feat*(n_feat-1)//2,count); plt.imshow(tmp_T_diff,
                                        interpolation = "none", vmin = -10, vmax = 10,
                                        extent = [times_ms[0], times_ms[-1], 0, n_channel],
                                    origin = "lower",aspect = "auto")
                            plt.title("%s-%s" %(feat_name_seq[j], feat_name_seq[i])); plt.colorbar()
                            plt.xlabel('time (ms)');
                            plt.ylabel('channel id')
                    plt.tight_layout()
                    #plt.savefig(fig_outdir + "AlexNet_%s_CCA_ncomp%d_%s_T_diff.pdf" %(MEGorEEG[isMEG], n_comp, data_name[ll]))
                    #plt.savefig(fig_outdir + "AlexNet_%s_CCA_ncomp%d_%s_T_diff.eps" %(MEGorEEG[isMEG], n_comp, data_name[ll]))
        """    
            
        cmap = plt.get_cmap('plasma') 
        #cmap = plt.get_cmap('jet')
        figsize = (13,6)
        for ll in range(len(data_to_plot)):
            for j in range(n_feat):
                tmp = data_to_plot[ll][:,j,:,:].mean(axis = 0)
                if Flag_PCA:
                    if Flag_additional:
                        figname = FIG_OUT_DIR + "%s_sensor_topo_%s_%s_ncomp%d_%s_%s_additional%d.pdf"  %(MEGorEEG[isMEG], feat_name_seq[j], 
                                            data_name[ll], n_comp, feature_suffix, fname_suffix, Flag_additional)
                    else:
                        figname = FIG_OUT_DIR + "%s_sensor_topo_%s_%s_ncomp%d_%s_%s_cv%d.pdf"  %(MEGorEEG[isMEG], feat_name_seq[j], 
                                            data_name[ll], n_comp, feature_suffix, fname_suffix, Flag_cv)
                        
                else:
                    if Flag_additional:
                        figname = FIG_OUT_DIR + "%s_sensor_topo_%s_%s_%s_%s_additional%d.pdf"  %(MEGorEEG[isMEG], feat_name_seq[j], 
                                            data_name[ll], feature_suffix, fname_suffix, Flag_additional )
                    else:
                        figname = FIG_OUT_DIR + "%s_sensor_topo_%s_%s_%s_%s_cv%d.pdf"  %(MEGorEEG[isMEG], feat_name_seq[j], 
                                            data_name[ll], feature_suffix, fname_suffix, Flag_cv)
                #visualize_sensors( tmp, vmin, vmax_list[ll], figname, cmap, figsize, isMEG, times1, time_array)
                if isMEG:
                    if feat_name1[j] != 'local contrast':
                        vmax = vmax_list[ll]*100 
                    else:
                        vmax = 14.0
                    visualize_sensors_MEG_topo( info1, tmp*100, vmin*100, vmax , figname, figsize, 
                      times1, time_array, title = feat_name1[j], ylim = [vmin*100, vmax_list[ll]*100])
                #plt.close('all')
        '''
        if not Flag_CCA and Flag_PCA:
            cmap1 = plt.get_cmap('seismic')         
            pairs = [[0,6]]
            n_pairs = len(pairs)
            Tvmax = 10
            for i in range(n_pairs):
                for ll in range(len(data_to_plot)):
                    tmp = data_to_plot[ll][:, pairs[i][1]] - data_to_plot[ll][:, pairs[i][0]]
                    tmpT = tmp.mean(axis = 0)/tmp.std(axis = 0)* np.sqrt(tmp.shape[0])
                    figname = fig_outdir + "%s_sensor_topo_%s-%s_ncomp%d_%sT_additional%d.pdf"  \
                             %(MEGorEEG[isMEG], feat_name_seq[pairs[i][1]], feat_name_seq[pairs[i][0]],
                                n_comp, data_name[ll], Flag_additional)
                    #visualize_sensors( tmpT, -Tvmax, Tvmax, figname, cmap1, figsize, isMEG,
                    #      times1, time_array)
                    if isMEG:
                        visualize_sensors_MEG_topo( info1, tmp, -Tvmax, Tvmax, figname, figsize, 
                          times, time_array, title = feat_name_seq[pairs[i][0]])
        '''
        '''
        if Flag_CCA and Flag_PCA:
            # difference between layer 1 and 6 T-tests
            figsize = (20,10) 
            cmap = plt.get_cmap('bwr') 
            tmp = Rsq[:,1,:,:]- Rsq[:,0,:,:]
            mean_tmp = tmp.mean(axis = 0)
            se_tmp = np.std(tmp, axis = 0)/np.sqrt(tmp.shape[0])
            T_mean_tmp = mean_tmp/se_tmp
            fig_name = fig_outdir + "Subj_pooled_sensor_topo_diff_Layer7_Layer1_%s_%s_cv%d.pdf"  \
                                          %(feature_suffix, fname_suffix, Flag_cv)
            vmin,vmax = -10,10
            #visualize_sensors(T_mean_tmp, vmin, vmax, fig_name, cmap, figsize, isMEG,
            #                  times1, time_array)
            if isMEG:
                visualize_sensors_MEG_topo( info1, T_mean_tmp,vmin, vmax, figname, figsize, 
                          times, time_array, title = "Layer7-Layer1")
            plt.close('all')
            # add permutation tests
        '''
        if not Flag_CCA and Flag_PCA: # plot the averaged significance of Rsq    
            NCURVES = len(feat_name_seq)
            values = range(NCURVES)
            jet = cm = plt.get_cmap('jet') 
            cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            # plot conv1, conv2, conv5 and fc6 in the same figure
            ymin, ymax = -4, 10
            ymin, ymax = None, None
            plt.figure(figsize = (5,5))
            ax = plt.subplot(1,1,1)
            count = 0
            alpha0 = 0.05
            threshold = scipy.stats.t.ppf(1-alpha0/2, n_Subj-1)
            cluster_p_thresh = 0.05/n_feat
            
            space = 0.5
            n_times = len(times)
                
            colorVal_list = list()
            for j in range(8):
                colorVal = scalarMap.to_rgba(values[j])
                colorVal_list.append(colorVal)
                
            tmp_f = lambda x:  "Layer%d" %x  
            layer_seq =  map( tmp_f, np.arange(1,9)) 
            for j in range(8):
                colorVal = colorVal_list[count]
                tmp_val_mean = (Rsq[:,j]).mean(axis = 1)*100.0
                ax.plot(times_ms, np.mean(tmp_val_mean, axis = 0), lw = 2, color = colorVal)
                tmp = bootstrap_mean_array_across_subjects(tmp_val_mean, alpha = 0.05/n_times/n_feat)
                tmp_mean = tmp['mean']
                tmp_se = tmp['se']
                ub = tmp['ub']
                lb = tmp['lb'] 
                _ = ax.fill_between(times_ms, ub, lb, facecolor=colorVal, alpha=0.15)
                
                baseline_time_ind = times< 0
                baseline_mean = np.mean(tmp_val_mean[:, baseline_time_ind], axis = 1)
                val_time_selected_no_baseline = (tmp_val_mean.T - baseline_mean).T
                Tobs, clusters, p_val_clusters, H0 = mne.stats.permutation_cluster_1samp_test(
                val_time_selected_no_baseline, threshold,tail = 1)
                print (clusters, p_val_clusters)
                tmp_window = list()
                for i_c, c in enumerate(clusters):
                    c = c[0]
                    if p_val_clusters[i_c] <= cluster_p_thresh:
                        #_ = ax.axvspan(times_ms[c.start], times_ms[c.stop - 1],color='r', alpha=0.2)
                        #_ = plt.text(times_ms[c.start],0.25,('p = %1.3f' %p_val_clusters[i_c])) 
                        tmp_window.append(dict(start = c.start, stop = c.stop, p = p_val_clusters[i_c]))
                        
                _ = ax.text(-85, -count*space, layer_seq[count], fontsize = 10, color = colorVal)            
                for i in range(len(tmp_window)): 
                    if tmp_window[i]['p'] <= cluster_p_thresh:
                        _= ax.plot(np.array([times_ms[tmp_window[i]['start']],  times_ms[tmp_window[i]['stop']]]),
                            np.array([(-count+0.5)*space, (-count+0.5)*space]), color = colorVal, lw = 2)
                
                
                count = count+1
                
    
            plt.xlabel('time (ms)')
            #plt.title("%s" %(MEGorEEG[isMEG]))
            plt.ylabel('% variance explained')
            plt.xlim(-100.0,900.0)
            plt.ylim(ymin, ymax)
            plt.tight_layout()
            fig_name = fig_outdir + \
                    "Subj_pooled_%s_multi_layers_%s_%s%d_Rsq_ave_sensor_cv%d" \
                    %(model_name, MEGorEEG[isMEG], 
                      feature_suffix, n_comp, Flag_cv)
            plt.savefig( fig_name+".pdf")
            #plt.savefig( fig_name+".png", dpi = 1000.0)





   
        """
            # test the response to width, height, apsect ratio and area
            mat_data = scipy.io.loadmat('/home/ying/Dropbox/Scene_MEG_EEG/Features/selected_image_second_round_data.mat');
            aspect_ratio = mat_data['aspect_ratio'] [:,0]
            n_im = len(aspect_ratio)
            # width, hight
            im_size = np.zeros([n_im, 2])  
            max_side = 500.0
            full_side = 600.0
            for i in range(n_im):
                if aspect_ratio[i] >= 1:
                    im_size[i,0] = max_side
                    im_size[i,1] = max_side/aspect_ratio[i]
                else:
                    im_size[i,0] = max_side*aspect_ratio[i]
                    im_size[i,1] = max_side
                    
            X = np.zeros([n_im,4])
            X[:,0:2] = im_size
            X[:,2] = aspect_ratio
            X[:,3] = im_size[:,0]*im_size[:,1]
            X = scipy.stats.zscore(X, axis = 0)
            log10p = np.zeros([n_Subj, n_channels, n_times])
            for i in range(n_Subj):
                t0 = time.time()
                subj = "Subj%d" %Subj_list[i]
                if isMEG:
                    ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
                else:
                    ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
                result_reg = sensor_space_regression_no_regularization(subj, ave_mat_path, X)
                log10p[i] = result_reg['log10p']
                times = result_reg['times']
            vmin, vmax = 0, 6
            figsize = (5, 4.5)
            plt.figure(figsize = figsize)
            times_ms = (times-offset)*1000
            plt.imshow(np.mean(log10p, axis = 0),
                        interpolation = "none", vmin = vmin, vmax = vmax,
                        extent = [times_ms[0], times_ms[-1], 0, n_channels],
                        origin = "lower", aspect = "auto")
            plt.colorbar()
            plt.xlabel('time (ms)');
            plt.ylabel('channel id')
            plt.savefig(fig_outdir + "%s_aspect_ave_logp.pdf" %(MEGorEEG[isMEG]))
            
        #======================================================================================
        # plot topological maps for each subject  no_permutation version

            suffix = "10_dim"

            Fval_all = np.zeros([n_Subj, n_feat, n_channels, n_times])
            logp_all = np.zeros([n_Subj, n_feat, n_channels, n_times])
    
            for j in range(n_feat):
                feat_name = feat_name_seq[j]
                for i in range(n_Subj):
                    subj = "Subj%d" % Subj_list[i]
                    if isMEG:
                        ave_mat_path = meg_dir + "%s/%s_%s.mat" %(subj,subj,fname_suffix)
                    else:
                        ave_mat_path = eeg_dir + "%s_EEG/%s_EEG_%s.mat" %(subj,subj,fname_suffix)
                    mat_name = mat_file_out_dir + "%s_%s_result_reg_%s_%s.mat"  %(subj, MEGorEEG[isMEG],
                                                                                  feat_name, suffix)  
                    tmp_data = scipy.io.loadmat(ave_mat_path)
                    tmp_picks = tmp_data['picks_all'][0].astype(np.int)
                    print len(tmp_picks)
                    tmp_data2 = scipy.io.loadmat(mat_name)
                    tmp_F_val = np.zeros([n_channels, n_times])
                    tmp_F_val[tmp_picks,:] = tmp_data2['Fval']
                    Fval_all[i,j] = tmp_F_val
                    
                    n_dim = tmp_data2['n_dim'][0]
                    print subj, n_dim
                    dfM, dfE = n_dim-1, n_im - n_dim
                    logp_all[i,j] = tmp_data2['log10p']
            times = tmp_data['times'][0]
            offset = 0.04 if isMEG else 0.00
            times_in_ms = (times -offset)*1000.0
            
            vmin, vmax = 0, 6
            figsize = (5, 4.5)
            for i in range(n_feat):
                plt.figure(figsize = figsize)
                plt.imshow(np.mean(logp_all[:,i],axis = 0), vmin = vmin, 
                           vmax = vmax, interpolation = "none", aspect = "auto",
                           extent = [times_in_ms[0], times_in_ms[-1], 1, n_channels], 
                           origin = "lower")
                plt.title(feat_name_seq[i])
                plt.colorbar() 
                plt.xlabel('time (ms)')
                plt.ylabel('channel ind')
                plt.savefig(fig_outdir + "Subj_pooled_%s_%s_ave_reg_%s_avep.pdf" %(feat_name_seq[i], MEGorEEG[isMEG],suffix))
                plt.close()
                    
        
        # plot the the difference of mean logp
        if False:
            figsize0 = (28, 24)
            plt.figure(figsize = figsize0)
            count = 0
            vabs = 1
            for i in range(n_feat_name):
                for j in range(i+1, n_feat_name):
                    plt.subplot(n_feat_name-1, n_feat_name-1, i*(n_feat_name-1) + j)
                    plt.imshow(np.mean(logp_all[:,i,:]- logp_all[:,j,:], axis = 1), aspect = "auto", 
                               interpolation = "none",
                               vmin = -vabs, vmax = vabs,
                               extent = [times_in_ms[0], times_in_ms[-1], 1, n_Subj], origin = "lower",)
                    
                    plt.title( "%s-%s" % (feat_name_seq[i],feat_name_seq[j]))
                    if j == i+1:
                        plt.xlabel('times (ms)')
                        plt.ylabel('subj id')
                    else:
                        plt.axis('off')
            plt.subplot(n_feat-1, n_feat-1, (n_feat-2)*(n_feat-1)+1 )
            plt.colorbar()
            plt.tight_layout(0.01)
            plt.savefig(fig_outdir + "Subj_pooled_%s_ave_reg_%s_avep_diff.pdf" %(MEGorEEG[isMEG],suffix)) 
            
            ind_conv1 = [l for l in range(n_feat) if feat_name_seq[l] in ['AlexNet_conv1_no_aspect']][0]
            ind_conv2 = [l for l in range(n_feat) if feat_name_seq[l] in ['AlexNet_conv2_no_aspect']][0]
            ind_conv5 = [l for l in range(n_feat) if feat_name_seq[l] in ['AlexNet_conv5_no_aspect']][0]
            ind_fc6 = [l for l in range(n_feat) if feat_name_seq[l] in ['AlexNet_fc6_no_aspect']][0]
            ind_fc7 = [l for l in range(n_feat) if feat_name_seq[l] in ['AlexNet_fc7_no_aspect']][0]                  
            # individual plot
            
            for i in [ind_conv1, ind_fc6]:
                for j in [ind_conv1, ind_fc6]:
                    if i == j:
                        continue
                    plt.figure(figsize = figsize)
                    plt.imshow(np.mean(logp_all[:,i,:]- logp_all[:,j,:], axis = 1), aspect = "auto", 
                               interpolation = "none",
                               vmin = -vabs, vmax = vabs,
                               extent = [times_in_ms[0], times_in_ms[-1], 1, n_Subj], origin = "lower",)
                    plt.title( "%s-%s" % (feat_name_seq[i],feat_name_seq[j]))
                    plt.xlabel('times (ms)')
                    plt.ylabel('subj id')
                    plt.colorbar()
                    plt.savefig(fig_outdir + "Subj_pooled_%s-%s_%s_ave_reg_%s_avep_diff.pdf" \
                              %(feat_name_seq[i],feat_name_seq[j], MEGorEEG[isMEG],suffix))
                    plt.close()
            
           
            
            # plot the t-statistics over all channels across subjects
            plt.figure(figsize = figsize0)
            count = 0
            vabs = 6
            for i in range(n_feat):
                for j in range(i+1, n_feat):
                    plt.subplot(n_feat-1, n_feat-1, i*(n_feat-1) + j)
                    tmp_diff = logp_all[:,i,:]- logp_all[:,j,:]
                    tmp_t = np.mean(tmp_diff, axis =0)/np.std(tmp_diff, axis = 0)*np.sqrt(n_Subj)
                    plt.imshow(tmp_t, aspect = "auto", interpolation = "none",
                               vmin = -vabs, vmax = vabs,
                               extent = [times_in_ms[0], times_in_ms[-1], 1, n_channels], origin = "lower",)
                    
                    plt.title( "%s-%s" % (feat_name_seq[i],feat_name_seq[j]))
                    
                    if j == i+1:
                        plt.xlabel('times (ms)')
                        plt.ylabel('channel id')
                    else:
                        plt.axis('off')
            
            plt.subplot(n_feat-1, n_feat-1, (n_feat-2)*(n_feat-1)+1 )
            plt.colorbar()
            plt.tight_layout(pad = 0.01)
            plt.savefig(fig_outdir + "Subj_pooled_%s_ave_reg_%s_diff_t_stat.pdf" %(MEGorEEG[isMEG],suffix))  
            
            # individual plots
            for i in [ind_conv1, ind_fc6]:
                for j in [ind_conv1, ind_fc6]:
                    if j == i:
                        continue
                    plt.figure(figsize = figsize)
                    tmp_diff = logp_all[:,i,:]- logp_all[:,j,:]
                    tmp_t = np.mean(tmp_diff, axis =0)/np.std(tmp_diff, axis = 0)*np.sqrt(n_Subj)
                    #tmp_p = 2.0*(1-scipy.stats.t.cdf(np.abs(tmp_t.ravel()), n_Subj-1))
                    #tmp_thresh,_ = mne.stats.fdr_correction(tmp_p, alpha = 0.1, method = "negcorr")
                    #tmp_thresh = np.reshape(tmp_thresh, tmp_t.shape)
                    #tmp_t[ np.reshape(tmp_p, tmp_t.shape)> 0.1] = 0
                    plt.imshow(tmp_t, aspect = "auto", interpolation = "none",
                               vmin = -vabs, vmax = vabs,
                               extent = [times_in_ms[0], times_in_ms[-1], 1, n_channels], origin = "lower",)
                    plt.title( "%s-%s" % (feat_name_seq[i],feat_name_seq[j]))
                    plt.xlabel('times (ms)')
                    plt.ylabel('channel id')
                    plt.colorbar()
                    plt.savefig(fig_outdir + "Subj_pooled_%s-%s_%s_ave_reg_%s_diff_t_stat.pdf" 
                    %(feat_name_seq[i],feat_name_seq[j], MEGorEEG[isMEG],suffix)) 
                    plt.close()
  
        # ==========topomap of the difference T, with  spatial temporal clustering tests
        if False:
            if isMEG:
                # this has gradiometers only
                #connectivity = mne.channels.read_ch_connectivity('neuromag306mag')
                layout = mne.channels.read_layout('Vectorview-all')
                pos3 = layout.pos
            else:
                pos3 = scipy.io.loadmat("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/biosemi128_layout/biosemi128_pos.mat")
                pos3 = pos3['pos']
                
            n_sensor = pos3.shape[0]   
            dist_mat = np.zeros([n_sensor,n_sensor])
            for i in range(n_sensor):
                for j in range(n_sensor):
                    dist_mat[i,j] = np.sqrt(np.sum((pos3[i,:] - pos3[j,:])**2) )
            
            if isMEG:
                connectivity = scipy.sparse.csr_matrix(dist_mat < 0.1*np.max(dist_mat))
            else:
                connectivity = scipy.sparse.csr_matrix(dist_mat < 0.15*np.max(dist_mat)) 
            
            
            tmp_diff = logp_all[:,ind_fc6,:]- logp_all[:,ind_conv1,:]
            T_thresh = None
            T_obs, clusters, cluster_p_values, H0 = \
            mne.stats.spatio_temporal_cluster_1samp_test(tmp_diff.transpose([0,2,1]), connectivity=connectivity, n_jobs=2,
                                       threshold=  T_thresh)
            p_accept = 0.05                          
            good_cluster_inds = np.where(cluster_p_values < p_accept)[0]
            vmin, vmax = -6,6
            for i_clu, clu_idx in enumerate(good_cluster_inds):
                # unpack cluster infomation, get unique indices
                time_inds, space_inds = np.squeeze(clusters[clu_idx])
                ch_inds = np.unique(space_inds)
                time_inds = np.unique(time_inds)
            
                # get topography for F stat
                f_map = T_obs[time_inds, ...].mean(axis=0)
                # create spatial mask
                mask = np.zeros((f_map.shape[0], 1), dtype=bool)
                mask[ch_inds, :] = True
            
                # initialize figure
                fig, ax_topo = plt.subplots(1, 1, figsize=(4, 3))
                title = "time %d to %d ms \n p = %1.3f" \
                        %(times_in_ms[time_inds.min()], times_in_ms[time_inds.max()], cluster_p_values[clu_idx])
                fig.suptitle(title, fontsize=16)
            
                # plot average test statistic and mark significant sensors
                image, _ = mne.viz.plot_topomap(f_map, pos3[:,0:2], mask=mask, axis=ax_topo,
                                         vmin=vmin, vmax=vmax, contours = False)
                plt.tight_layout()
                # advanced matplotlib for showing image with figure and colorbar
                # in one plot
                divider = make_axes_locatable(ax_topo)
            
                # add axes for colorbar
                ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
                plt.colorbar(image, cax=ax_colorbar)
                
                fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_reg/"
                plt.savefig(fig_outdir + "Subj_pooled_conv1fc6_%s_%s_log10p_mean_sensor_topo_Cluster%d.pdf" \
                            %(MEGorEEG[isMEG], suffix, i_clu))
                plt.savefig(fig_outdir + "Subj_pooled_conv1fc6_%s_%s_log10p_mean_sensor_topo_Cluster%d.eps" \
                            %(MEGorEEG[isMEG], suffix, i_clu))            
            
            """
        
        
            
    

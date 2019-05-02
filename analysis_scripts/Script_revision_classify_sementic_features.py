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

path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
#path1 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
path1 = "/media/yy/LinuxData/yy_dropbox/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path1)
#path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
path0 = "/media/yy/LinuxData/yy_dropbox/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from Stat_Utility import bootstrap_mean_array_across_subjects

#=======================================================================================




#=======================================================================================
if __name__ == "__main__":
    
    dropbox_dir = "/media/yy/LinuxData/yy_dropbox/Dropbox/"
    data_root_dir = "/media/yy/LinuxData/yy_Scene_MEG_data/MEG_preprocessed_data/MEG_preprocessed_data/"   
    model_name = "AlexNet"
    n_im = 362
    
    t_start = 300.0
    t_end = 380.0
    C = 10.0
    
    import time
    np.random.seed(np.int(time.time()))
    
    #%% load the regression design matrix
    
    #regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
    #mat_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/ave_ols/"
    #fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_reg/MEEG_no_aspect/"
    
    regressor_dir = "/media/yy/LinuxData/yy_Scene_MEG_data/regressor/"
    
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
            

    # add another set, the top n_compnent 
    fname =  "/media/yy/LinuxData/yy_dropbox/Dropbox/" + \
        "/Scene_MEG_EEG/Features/" + \
        "Layer1_contrast/Stim_images_Layer1_contrast_noaspect.mat"
    mat_dict = scipy.io.loadmat(fname)
    tmp = mat_dict['contrast_noaspect']
    tmp = tmp- np.mean(tmp, axis = 0)
    U,D,V = np.linalg.svd(tmp)
    var_prop = np.cumsum(D**2)/np.sum(D**2)
    
    X_list.append(U[:,0:n_comp])
    feat_name_seq.append("contrast")
    n_feat = len(X_list)

    
    
    
    """
    # add the data from V1 from 300-350 ms, use lambda = 1.0, for each subjects
    """
    

    data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
    data_root_dir =data_root_dir0 +"MEG_preprocessed_data/MEG_preprocessed_data/"
    lambda2 = 1.0
    MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    subj_list = range(1, 19)
    #times = np.arange(0.01,1.01,0.01)
    #n_times = 100
    #time_in_ms = (times - 0.04)*1000.0
    
    ROI_name = 'pericalcarine'               
    labeldir = data_root_dir + "/ROI_labels/"  
    
    # TBA
    n_subj = len(subj_list)
    stc_out_dir = data_root_dir0 + "Result_Mat/" \
                + "source_solution/dSPM_MEG_ave_per_im/"
    fname_suffix = MEG_fname_suffix
    lambda2 = 1.0
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        labeldir1 = labeldir + "%s/labels/" % subj
        # load and merge the labels
        labels_bihemi = list()
        tmp_name = ROI_name
        #TBA
        tmp_label_list = list()
        for hemi in ['lh','rh']:
            print subj, tmp_name, hemi
            tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,tmp_name,hemi)
            tmp_label = mne.read_label(tmp_label_path)
            tmp_label_list.append(tmp_label)
        labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1]) 
        fwd_path = data_root_dir + "/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
            
        fwd = mne.read_forward_solution(fwd_path) 
        fwd = mne.convert_forward_solution(fwd, surf_ori = True,)
        src = fwd['src']       
        tmp_label = labels_bihemi[0]
        _, ROI_ind = mne.source_space.label_src_vertno_sel(tmp_label, src)

        print ("number of sources %d" %(len(ROI_ind)))
        # load the source solution
        mat_dir = stc_out_dir + "%s_%s_%s_lambda2_%1.1f_ave.mat" %(
                subj, "MEG",fname_suffix, lambda2)
        mat_dict = scipy.io.loadmat(mat_dir)
        source_data = mat_dict['source_data']
        times = mat_dict['times'][0]
        time_in_ms = (times - 0.04)*1000.0
        del(mat_dict)
            
        
        
        tmp = source_data[:,ROI_ind,:]
        tmp -= np.mean(tmp, axis = 0)
        
        
        
        t_ind = np.all(np.vstack([time_in_ms> t_start,
                                  time_in_ms < t_end]),
                       axis = 0)
        
        X = tmp[:,:,t_ind].mean(axis = 2)
        X = X/np.std(X, axis = 0)
        
        '''
        # if applying SVD for fair comparison
        U,D,V = np.linalg.svd(X, full_matrices = False)
        U = U[:,0:6]
        print (U.shape)
        '''
        
        
        
        X_list.append(U)
        del(source_data) 
        
        
    feat_name_seq_show = ['res Layer 1','res Layer 7',
                          'common','local contrast']
    for i in range(n_subj):
        feat_name_seq_show.append("Participant %d" %(i+1))

    n_features = len(X_list)
    
    
    
    
    #%% load the csv
    csv_name = dropbox_dir + "Scene_MEG_EEG/Features/"+ \
        "MEG_NEIL_Image_SUN_hierarchy - editedfromsun.csv"
    '''   
    csv_name = dropbox_dir + "Scene_MEG_EEG/Features/"+ \
        "MEG_NEIL_Image_SUN_hierarchy - Genereated_from_SUN_with_Comments.csv"
    '''
    
    with open(csv_name,'r') as fid:
        lines = fid.readlines()
    
    column_names = lines[0].split(",")
    lines.pop(0)
    
    semantic_feat = np.zeros([362,19])
    # hard coded
    for (i, line) in enumerate(lines):
        tmp_lines = line.split("," or "\n")
        for j in range(1, 20):
            semantic_feat[i,j-1] = np.int(tmp_lines[j])
            
    # print the class number for the first three cateogories
    print ( np.sum(semantic_feat, axis = 0) )
        
    #%%
    # try a binary classification for the first classes
    # to ROC with cross validation
    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import StratifiedKFold
    from scipy import interp

    area_under_curve = np.zeros([3,len(X_list)])
    area_under_curve1 = np.zeros([3,len(X_list)])
    roc_curves = np.zeros([3, len(X_list)], dtype = np.object)
    random_state = np.random.RandomState(0)
    n_splits = 10
    flag_plot = True
    figsize = (5,3)
    fig_out_dir = "/media/yy/LinuxData/yy_Scene_MEG_data/Result_Mat/" + \
        "figs/feature_analysis/"
    
    # example
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    classification_task_names = ['indoor','outdoor_natrual','outdoor_man-made']
    mean_fpr = np.linspace(0, 1, 100)
    
    for classification_task_column in range(3):
        # indoor and outdoor
        cv = StratifiedKFold(n_splits=n_splits)
        classifier = svm.SVC(kernel='linear', probability=True,
                     random_state=random_state, C = C)
 
        y = semantic_feat[:,classification_task_column]
        for (l,X) in enumerate(X_list):
            #l = 2
            #X = X_list[l]
            n_samples, n_features = X.shape
            
            tprs = []
            aucs = []
            #mean_fpr = np.linspace(0, 1, 100)

            for train, test in cv.split(X, y):
                probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
                # Compute ROC curve and area the curve
                fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

            
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            
            area_under_curve[classification_task_column, l] = np.mean(aucs)
            area_under_curve1[classification_task_column, l] = mean_auc
            roc_curves[classification_task_column, l] = mean_tpr
            
            '''
            if flag_plot:
                fig = plt.figure(figsize = figsize)
                plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                     label='Chance', alpha=.8)
            
            
                plt.plot(mean_fpr, mean_tpr, color='b',
                         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                         lw=2, alpha=.8)
                
                std_tpr = np.std(tprs, axis=0)
                tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                                 label=r'$\pm$ 1 std. dev.')
            
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.title("feature = %s, classifying %s" %(feat_name_seq[l],
                    classification_task_names[classification_task_column]))
                
                fig.savefig(fig_out_dir + "cv_ROC_%s_%s.eps" %(feat_name_seq[l],
                    classification_task_names[classification_task_column]))
                plt.close('all')
             '''   
 #%%           
if flag_plot:
    
    figsize = (18,5)
    fig = plt.figure(figsize = figsize)
    n_task = 3
    for classification_task_column in range(n_task):
        plt.subplot(1,n_task, classification_task_column+1)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k',
             label='Chance', alpha=.5)
        color_seq = np.array(['b','g','r','k',])
        #color_seq = np.hstack( [color_seq, np.tile(['y'],  n_subj)])
        for (l,X) in enumerate(X_list[0:4]):
            plt.plot(mean_fpr, 
                     roc_curves[classification_task_column,l],
                     color=color_seq[l],
                     label=r'%s (AUC = %0.2f )'\
                     % (feat_name_seq_show[l],
                        area_under_curve1[classification_task_column, l]),
                     lw=2, alpha=.8)
                     
        # plot the bootstrapped confidence interval
        tmp_data = np.array(roc_curves[classification_task_column,4::].tolist())
        btstrp_alpha =  0.05/tmp_data.shape[1]
        tmp = bootstrap_mean_array_across_subjects(tmp_data, alpha = btstrp_alpha)
        tmp_mean = tmp['mean']
        tmp_se = tmp['se']
        ub = tmp['ub']
        lb = tmp['lb'] 
        
        plt.plot(mean_fpr, tmp_mean, color = 'y', lw = 3, alpha = 0.9,
                 label=r'EVC %d~%d ms (AUC = %0.2f )' \
                         %(t_start,t_end,
                    area_under_curve1[classification_task_column, 4:].mean()))
        plt.fill_between(mean_fpr, lb, ub, alpha = 0.5, color = 'y',
                         )
        
        
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(classification_task_names[classification_task_column] )
        plt.legend(loc="lower right")
                
    fig.savefig(fig_out_dir + "cv_ROC_feature_semantic_SVMC%1.1f.pdf" %C)
    fig.savefig(fig_out_dir + "cv_ROC_feature_semantic_SVMC%1.1f.png" %C, 
                 dpx = 500)
    #plt.close('all')


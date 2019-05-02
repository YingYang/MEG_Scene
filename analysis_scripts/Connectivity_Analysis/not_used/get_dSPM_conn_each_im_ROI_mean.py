import numpy as np
import sys
import scipy.io

path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Connectivity_Analysis/"
sys.path.insert(0, path0)
from conn_utility import (get_tf_PLV, get_stationary_spec_conn, get_corr_tf)
path1 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path1)
from Stat_Utility import Fisher_method
import scipy.stats

#================================================================================
def get_conn_each_im(subj, ind_array_tuple, ROI_ts_fname, ROI_ind, method = "plv", 
                     demean = False, bin_size = None):
    """
    Input:
        subj = Subj1
        ind_array_tuple = (array1, array2)
             pairs are (array1[i], array2[i])
        ROI_ts_fname, full path of the mat file, where the ROI ts are saved
        ROI_ind, the list of indices of ROIs in the aparc label set
        method = "plv", phase locking value (time freq)
               = "corr", correlation of the STFT components across trials
        demean: flag for all methods other than "corr"
                if True, remove the mean
        bin_size, if an integer, take the mean within a bin size of the integer
                    if None, do nothing
    Output:
        conn_list, 361 connectivity matrices for each image, each matrix is [n_pairs, n_freqs, n_times]
        freqs, array of frequence 
        times_conn, times of the conn matrix
    """
    mat_dict = scipy.io.loadmat(ROI_ts_fname)
    #  mat_dict = dict(ROI_ts_each_trial = data_mat_no_repeat, im_id = im_id_no_repeat,
    #                ROI_names = ROI_names, times = times,
    #                epochs_fname = epochs_fname)
    ROI_ts_each_trial = mat_dict['ROI_ts_each_trial']
    ROI_names = mat_dict['ROI_names']
    times = mat_dict['times'][0]
    im_id = mat_dict['im_id'][0]
    n_im = np.int(np.max(im_id)+1)
    ROI_ts_each_trial_sel = ROI_ts_each_trial[:, ROI_ind]
    
    conn_list = list()
    for i in range(n_im):
        tmp_data = ROI_ts_each_trial_sel[im_id == i]
        if method == "plv":
            tmp, freqs, times_conn = get_tf_PLV(tmp_data, ind_array_tuple, demean = demean,
                                         time_start = times.min(), sfreq  = 1000.0)
        elif method == "corr":
            tmp_cov, tmp_corr, freqs, tstep, wsize = get_corr_tf(tmp_data, 
                                    sfreq = 1000.0, wsize = 160, tstep = 40)
            tmp_corr1 = np.mean(np.abs(tmp_corr),axis = 0) # note frequency 0 is half   
            tmp_corr1[0] = np.abs(tmp_corr[0,0]) 
            tmp = (tmp_corr1[:,:, ind_array_tuple[0], ind_array_tuple[1]]).transpose([2,0,1])   
            if i == 0:
                times_conn = times[0:-1:tstep]+times[tstep//2]
        if bin_size is not None:
            n_times = len(times_conn)
            tmp_new = np.reshape(tmp[:,:,0:n_times//bin_size*bin_size], 
                                 [-1, len(freqs), n_times//bin_size, bin_size])
            tmp = np.mean(tmp_new, axis = -1)
            times_conn = (np.reshape(times_conn[0:n_times//bin_size*bin_size],
                                    [n_times//bin_size, bin_size])).mean(axis = -1)    
        conn_list.append(tmp) 
    return conn_list, freqs, times_conn                                
    
#==============================================================================
   
if __name__ == "__main__":
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    subj_list = range(1,14)
    n_run_per_subj_MEG = [6,12,6,12,10,8,12,10,10,10,12,12,12]
    n_subj = len(subj_list)
    isMEG = True
    
    label_set_name = "sceneROI"
    ROI_aggre_mode = "mean_flip"
    #ROI_name_list = ["parahippocampal-lh", "parahippocampal-rh",
    #            "medialorbitofrontal-lh", "medialorbitofrontal-rh",
    #            "pericalcarine-lh", "pericalcarine-rh",
    #            "lateraloccipital-lh","lateraloccipital-rh" ]
    ROI_names = [ 'medialorbitofrontal-lh', 'medialorbitofrontal-rh',
                        'pericalcarine-lh', 'pericalcarine-rh',
                        'lateraloccipital-lh','lateraloccipital-rh',
                        'PPA_c_g-lh', 'PPA_c_g-rh',
                        'TOS_c_g-lh', 'TOS_c_g-rh',
                        'RSC_c_g-lh', 'RSC_c_g-lh']
    # a shorter ROI name plot titles                    
    ROI_names_plot = ['mobf-l','mobf-r','v1-l','v1-r','lo-l','lo-r',
                      'PPA-l','PPA-r','TOS-l','TOS-r','RSC-l','RSC-r']                    
    nROI = len(ROI_names)          
    ROI_set_name = "sceneROI_mPFC_LO_EVC"
    # for now it is hand coded
    ROI_ind = range(nROI)
    ind_array_tuple = ([0,1,0,1, 0, 1, 2,3,2,3, 2, 3,  2,3, 0,1, 6,7, 6, 7, 8,9],
                       [6,7,8,9,10,11, 6,7,8,9,10,11,  4,5, 4,5, 8,9,10,11,10,11])
    n_im = 362
    offset = 0.04
    method = "plv"
    demean = False
    bin_size = 10
    outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/"

    #=================== compute the connectivity =============================
    if True: 
        print "%d subjects" %n_subj
        for i in range(n_subj):
            subj= "Subj%d" %subj_list[i]
            savedir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"+\
                            "Result_MAT/source_solution/dSPM_MEG_ROI_single_trial/%s/" %subj 
            ROI_ts_fname =  savedir + "%s_MEG_%s_%s_ROI_ts.mat" %(subj, label_set_name, ROI_aggre_mode)
            conn_list, freqs, times_conn = get_conn_each_im(subj, ind_array_tuple, 
                                            ROI_ts_fname, ROI_ind, method = method,
                                            demean = demean, bin_size = bin_size)
            conn_mat = np.zeros([n_im, len(ind_array_tuple[0]), len(freqs), len(times_conn)])
            for k in range(n_im):
                conn_mat[k] = conn_list[k] 
            mat_dict = dict(conn_mat = conn_mat, times_conn = times_conn, freqs = freqs,
                        ROI_names = ROI_names, ROI_ind = ROI_ind, ind_array_tuple = ind_array_tuple)
            mat_name =  outdir + "%s_MEG_%s_%s_demean%d.mat" %(subj,ROI_set_name, method, demean)
            scipy.io.savemat(mat_name, mat_dict)
    if False:
        n_pairs = len(ind_array_tuple[0])
        n_freqs, n_times = 95,110
        conn_mat_all_subj = np.zeros([n_subj, n_im, n_pairs, n_freqs, n_times])
        for i in range(n_subj):
            subj= "Subj%d" %subj_list[i]
            mat_name =  outdir + "%s_MEG_%s_%s_demean%d.mat" %(subj,ROI_set_name, method, demean)
            mat_dict = scipy.io.loadmat(mat_name)
            conn_mat_all_subj[i] = mat_dict['conn_mat']
        
        ind_array_tuple = mat_dict['ind_array_tuple']
        times_conn = mat_dict['times_conn'][0]
        freqs = mat_dict['freqs'][0]
        ROI_name_list_u = mat_dict['ROI_names']
  
        times_ms = (times_conn -offset)*1000
        
        # conn_mat_all_subj, dimensions, [n_subj, n_im, n_pairs, n_freqs n_times]
        fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/" 
        #============== Test 0, comparison with the baseline =====================
        # general sum across all subjects, with no tests at all
        mean_conn_mat = (conn_mat_all_subj.mean(axis = 1).mean(axis = 0))
        n1,n2 = 4,6
        plt.figure()
        for l in range(len(ind_array_tuple[0])):
            _=plt.subplot(n1,n2,l+1)
            _=plt.imshow(mean_conn_mat[l],aspect = "auto", interpolation = "none", vmin = 0, vmax = 1)
            _=plt.title( ROI_names_plot[ind_array_tuple[0][l]]+" " 
                 +ROI_names_plot[ind_array_tuple[1][l]] )
            _=plt.colorbar()
        # take the mean before time < 0 off 
        # this will give memory error
        #conn_mat_no_baseline = np.transpose(conn_mat_all_subj.transpose([4,0,1,2,3]) 
        #        - np.mean(conn_mat_all_subj[:,:,:,:,times_conn<= offset], axis = -1),[1,2,3,4,0])
        # try a different way
        for i in range(n_subj):
            for j in range(n_im):
                tmp = conn_mat_all_subj[i,j]
                tmp = np.transpose(tmp,[2,0,1]) - np.mean(tmp[:,:,times_conn<=offset], axis = -1)
                conn_mat_all_subj[i,j] = tmp.transpose([1,2,0])
        conn_mat_no_baseline = conn_mat_all_subj 
        T_increase = np.mean(conn_mat_no_baseline, axis = 1)/np.std(conn_mat_no_baseline, axis = 1)*np.sqrt(n_im)
        mean_T_increase = T_increase.mean(axis = 0)
        # do Fisher's method here?
        # [right sided, left sided]
        p_indiv_T_increase = [ (1-scipy.stats.t.cdf(T_increase,df = n_im -1)),
                              (scipy.stats.t.cdf(T_increase,df = n_im -1))]
         
        if True:
            plt.figure()
            for i in range(n_subj):
                _= plt.subplot(5,5,i+1)
                #plt.imshow(-np.log10(p_indiv_T_increase[0][i,0]), 
                #           aspect = "auto", interpolation = "none")
                _= plt.imshow(T_increase[i,6], 
                           aspect = "auto", interpolation = "none", vmin = -6, vmax = 6)
                _=plt.colorbar()                             
                              
        n_pair_ROI, n_freqs, n_times = len(ind_array_tuple[0]), len(freqs), len(times_ms)
        # right and left
        logp_Fisher = np.zeros([2, n_pair_ROI, n_freqs, n_times])
        for k0 in range(2):
            for k1 in range(n_pair_ROI):
                for k2 in range(n_freqs):
                    for k3 in range(n_times):
                        logp_Fisher[k0,k1,k2,k3] = -np.log10(
                        Fisher_method(p_indiv_T_increase[k0][:,k1,k2,k3], axis = 0))            
        
        data = [mean_conn_mat, mean_T_increase, logp_Fisher[0], logp_Fisher[1]]
        data_names = ['mean conn', 'mean T', 'logp Fisher right', 'logp Fisher left']
        vmin_seq = [0.3,-3, 0,0]
        vmax_seq = [0.7, 3, 6,6]
        for l in range(4):
            plt.figure(figsize = (25,12))
            for k in range(len(ind_array_tuple[0])):
                _= plt.subplot(n1,n2,k+1)
                _=plt.imshow(data[l][k], aspect = "auto", interpolation = "none",
                           extent = [times_ms[0],times_ms[-1],freqs[0], freqs[-1]],
                           origin = "lower",
                           vmin = vmin_seq[l], vmax = vmax_seq[l]);
                _=plt.colorbar(); 
                _=plt.title(ROI_names_plot[ind_array_tuple[0][k]]+"\n" 
                 +ROI_names_plot[ind_array_tuple[1][k]] )
                _=plt.xlabel('time (ms)')
                _=plt.ylabel('frequency (Hz)')
            plt.tight_layout()
            plt.savefig(fig_outdir + "conn_no_baseline_%s_%s_demean%d.pdf" %(method, data_names[l], demean)) 
            plt.close()
        #============== Test1, within category comparison of high-low attribute ===
        # put the image index into a [181,2] matrix!!
        # load 
        mat_data = scipy.io.loadmat('/home/ying/dropbox_unsync/MEG_scene_neil/PTB_Experiment/selected_image_second_round_data.mat');
        neil_attr_score = mat_data['attr_score']
        neil_low_level = mat_data['low_level_feat']
        is_high = mat_data['is_high'][:,0]
        n_cat = 181
        cat_label = np.ravel(np.tile(np.arange(0, n_cat),[2,1]).T)
        cat_hilo_table = np.zeros([n_cat, 2])
        for i in range(n_cat):
            tmp = np.nonzero(cat_label == i)[0]
            cat_hilo_table[i,0] = tmp[np.nonzero(is_high[tmp] == 1)[0]]
            cat_hilo_table[i,1] = tmp[np.nonzero(is_high[tmp] != 1)[0]]
        cat_hilo_table = cat_hilo_table.astype(np.int)
        # take the difference between high and low
        conn_all_subj_hilo = conn_mat_no_baseline[:,cat_hilo_table[:,0]] -conn_mat_no_baseline[:,cat_hilo_table[:,1]]
        # t-test where each category is a sample 
        conn_all_subj_hilo = conn_all_subj_hilo[1::]
        conn_all_subj_T_hilo = np.mean(conn_all_subj_hilo, axis = 1)/np.std(conn_all_subj_hilo, axis = 1)*np.sqrt(n_cat)
        p_indiv_T_hilo =[(1-scipy.stats.t.cdf((conn_all_subj_T_hilo),df = n_cat -1)),
                         scipy.stats.t.cdf((conn_all_subj_T_hilo),df = n_cat -1)]
        logp_Fisher_hilo = np.zeros([2,n_pair_ROI, n_freqs, n_times])
        for k0 in range(2):
            for k1 in range(n_pair_ROI):
                for k2 in range(n_freqs):
                    for k3 in range(n_times):
                        logp_Fisher_hilo[k0,k1,k2,k3] = -np.log10(Fisher_method
                        (p_indiv_T_hilo[k0][:,k1,k2,k3], axis = 0))            
        
        mean_T_hilo = np.mean(conn_all_subj_T_hilo, axis = 0)
        data_diff = [mean_T_hilo, logp_Fisher_hilo[0], logp_Fisher_hilo[1]]
        data_diff_names = ['aveT','logpFisher right', 'logpFisher left']
        vmin_seq = [-2, 0,0]
        vmax_seq = [ 2, 6,6]
        for l in range(3):
            plt.figure(figsize = (25,9))
            for k in range(len(ind_array_tuple[0])):
                _= plt.subplot(n1,n2,k+1)
                plt.imshow(data_diff[l][k], aspect = "auto", interpolation = "none",
                           extent = [times_ms[0],times_ms[-1],freqs[0], freqs[-1]],
                           origin = "lower", vmin = vmin_seq[l], vmax = vmax_seq[l]);
                plt.colorbar(); 
                plt.title(ROI_names_plot[ind_array_tuple[0][k]]+"\n" 
                +ROI_names_plot[ind_array_tuple[1][k]] )  
                plt.xlabel('time (ms)')
                plt.ylabel('frequency (Hz)')
            plt.tight_layout()
            plt.savefig(fig_outdir + "conn_no_hilo_%s_%s_demean%d.pdf" %(method, data_diff_names[l], demean)) 
            plt.close()
        
        #============== Test2, regresssion against the neil attributes/Alex feats===                     
        feat_name = "hybridCNN_conv5"
        var_percent = 15   
        regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"
        regressor_fname = regressor_dir +  "%s_PCA.mat" %(feat_name)
        tmp = scipy.io.loadmat(regressor_fname)
        X0 = tmp['X']
        if var_percent > 1:
            n_dim = np.int(min(X0.shape[1], var_percent))
            suffix = "%d_dim" % n_dim
        else:
            n_dim = np.nonzero(tmp['var_per_cumsum'][0] >=var_percent)[0][0]
            suffix = "%d_percent" % np.int(var_percent*100)
        
        print suffix, n_dim
        X = X0[:,0:n_dim]
        X = X- np.mean(X, axis = 0) 
        # load ols regression module
        path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
        sys.path.insert(0, path0)
        from ols_regression import ols_regression
        n_im = 362
        log10p = np.zeros([n_subj, n_pair_ROI, n_freqs, n_times])
        for i in range(n_subj):
            print i
            print n_subj
            tmp_conn = conn_mat_no_baseline[i].reshape([n_im, n_pair_ROI, -1])
            tmp_stat = ols_regression(tmp_conn, X, stats_model_flag = False)
            log10p[i] = tmp_stat['log10p'].reshape([n_pair_ROI, n_freqs, n_times])
            print log10p[i][0,0,0:5]
        
        mean_log10p = log10p.mean(axis = 0)
        log10p_Fisher_reg = np.zeros([n_pair_ROI, n_freqs, n_times])
        for k1 in range(n_pair_ROI):
            for k2 in range(n_freqs):
                for k3 in range(n_times):
                    log10p_Fisher_reg[k1,k2,k3] = -np.log10(Fisher_method(10.0**(-log10p[:,k1,k2,k3]), axis = 0))            
  
        data_reg = [mean_log10p, log10p_Fisher_reg]
        data_reg_names = ['ave_logp','logpFisher']
        vmin_seq = [0,0]
        vmax_seq = [3,6]
        for l in range(2):
            plt.figure(figsize = (25,12))
            for k in range(n_pair_ROI):
                _= plt.subplot(n1,n2,k+1)
                plt.imshow(data_reg[l][k], aspect = "auto", interpolation = "none",
                           extent = [times_ms[0],times_ms[-1],freqs[0], freqs[-1]],
                           origin = "lower", vmin = vmin_seq[l], vmax = vmax_seq[l]);
                plt.colorbar(); 
                plt.title(ROI_names_plot[ind_array_tuple[0][k]]+"\n" 
                +ROI_names_plot[ind_array_tuple[1][k]] )
                plt.xlabel('time (ms)')
                plt.ylabel('frequency (Hz)')
            plt.tight_layout()
            plt.savefig(fig_outdir + "conn_reg_%s_%s_%s_demean%d.pdf" \
                      %(data_reg_names[l],feat_name, method, demean)) 
            plt.close()
    
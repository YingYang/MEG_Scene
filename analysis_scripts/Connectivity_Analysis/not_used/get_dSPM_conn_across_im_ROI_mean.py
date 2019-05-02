import numpy as np
import sys
import scipy.io
import mne

import matplotlib
matplotlib.use('Agg')

path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Connectivity_Analysis/"
sys.path.insert(0, path0)
from conn_utility import (get_tf_PLV, get_corr_tf)
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
    data = ROI_ts_each_trial[:, ROI_ind]
    # substract the mean off
    data -= data.mean(axis = 0)
    
    if method == "plv":
        tmp, freqs, times_conn = get_tf_PLV(data, ind_array_tuple, demean = demean,
                                     time_start = times.min(), sfreq  = 1000.0)
    elif method == "corr":
        tmp_cov, tmp_corr, freqs, tstep, wsize = get_corr_tf(data, 
                                sfreq = 1000.0, wsize = 160, tstep = 40)
        tmp_corr1 = np.mean(np.abs(tmp_corr),axis = 0) # note frequency 0 is half   
        tmp_corr1[0] = np.abs(tmp_corr[0,0]) 
        tmp = (tmp_corr1[:,:, ind_array_tuple[0], ind_array_tuple[1]]).transpose([2,0,1])   
        times_conn = times[0:-1:tstep]+times[tstep//2]
    if bin_size is not None:
        n_times = len(times_conn)
        tmp_new = np.reshape(tmp[:,:,0:n_times//bin_size*bin_size], 
                             [-1, len(freqs), n_times//bin_size, bin_size])
        tmp = np.mean(tmp_new, axis = -1)
        times_conn = (np.reshape(times_conn[0:n_times//bin_size*bin_size],
                                [n_times//bin_size, bin_size])).mean(axis = -1)    
    conn = tmp 
    return conn, freqs, times_conn                                
    
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
    offset = 0.04
    #method = "plv"
    #bin_size = 10
    method = "corr"
    bin_size = None
    #=================== compute the connectivity =============================
    if True: 
        print "%d subjects" %n_subj
        conn_all_subj = list()
        for i in range(n_subj):
            subj= "Subj%d" %subj_list[i]
            savedir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"+\
                            "Result_MAT/source_solution/dSPM_MEG_ROI_single_trial/%s/" %subj 
            ROI_ts_fname =  savedir + "%s_MEG_%s_%s_ROI_ts.mat" %(subj, label_set_name, ROI_aggre_mode)
            conn, freqs, times_conn = get_conn_each_im(subj, ind_array_tuple, 
                                            ROI_ts_fname, ROI_ind, method = method,
                                            demean = False, bin_size = bin_size)            
            conn_all_subj.append(conn)
        
        conn_mat_all_subj = np.zeros(np.hstack([n_subj, conn_all_subj[0].shape]))
        for i in range(n_subj):
            conn_mat_all_subj[i] = conn_all_subj[i]
            
        mat_dict = dict(conn_mat_all_subj = conn_mat_all_subj, times_conn = times_conn, freqs = freqs,
                        ROI_names = ROI_names, ROI_ind = ROI_ind, ind_array_tuple = ind_array_tuple,
                        subj_list = subj_list)
        outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/"
        mat_name =  outdir + "all_subj_MEG_%s_%s_across_im.mat" %(ROI_set_name, method)
        scipy.io.savemat(mat_name, mat_dict)
        
    if True:   
        #============== analysis of the saved data ===============================
        outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_connectivity/"
        mat_name =  outdir + "all_subj_MEG_%s_%s_across_im.mat" %(ROI_set_name, method)
        mat_dict = scipy.io.loadmat(mat_name)
        
        conn_mat_all_subj = mat_dict['conn_mat_all_subj']
        ind_array_tuple = mat_dict['ind_array_tuple']
        times_conn = mat_dict['times_conn'][0]
        freqs = mat_dict['freqs'][0]
        ROI_name_list_u = mat_dict['ROI_names']
        subj_list = mat_dict['subj_list'][0]
        n_subj = len(subj_list)
        print "%d subjects" %n_subj
        
        times_ms = (times_conn -offset)*1000
        # conn_mat_all_subj, dimensions, [n_subj, n_im, n_pairs, n_freqs n_times]
        fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/" 
        #============== Test 0, comparison with the baseline =====================
        # general sum across all subjects, with no tests at all
        mean_conn_mat = conn_mat_all_subj.mean(axis = 0)
        n1,n2 = 4,6
        vmin, vmax = None,None
        plt.figure()
        for l in range(len(ind_array_tuple[0])):
            _=plt.subplot(n1,n2,l+1)
            _=plt.imshow(mean_conn_mat[l],aspect = "auto", interpolation = "none", vmin = vmin, vmax = vmax)
            _=plt.title( ROI_names_plot[ind_array_tuple[0][l]]+" " 
                 +ROI_names_plot[ind_array_tuple[1][l]] )
            _=plt.colorbar()
       
        for i in range(n_subj):
            tmp = conn_mat_all_subj[i]
            tmp = np.transpose(tmp,[2,0,1]) - np.mean(tmp[:,:,times_conn<=offset], axis = -1)
            conn_mat_all_subj[i] = tmp.transpose([1,2,0])
        conn_mat_no_baseline = conn_mat_all_subj 
        T_increase = np.mean(conn_mat_no_baseline, axis = 0)/np.std(conn_mat_no_baseline, axis = 0)*np.sqrt(n_subj)
        # do Fisher's method here?
        # [right sided, left sided]
        p_T = 2*(1-scipy.stats.t.cdf(np.abs(T_increase),df = n_subj -1))
        
        # try BH-FDR
        reject, _ = mne.stats.fdr_correction(p_T.ravel(),alpha = 0.05)

        data = [mean_conn_mat, T_increase]
        data_names = ['mean conn', 'mean T']
        plt.figure(figsize = (25,12))
        vmin_seq = [None,-6, 0,0]
        vmax_seq = [None, 6, 4,4]
        for l in range(2):
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
            plt.savefig(fig_outdir + "conn_no_baseline_%s_%s_across_im.pdf" %(method, data_names[l])) 
            plt.close()

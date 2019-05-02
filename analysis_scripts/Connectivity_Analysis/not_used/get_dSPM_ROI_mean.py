import numpy as np
import scipy.io
import mne
#=============================================================================
def get_dSPM_ROI_mean(subj, epochs_fname, run_id, fwd, outdir,
                             labels, label_names, label_set_name, 
                             ROI_aggre_mode = "mean_flip", isMEG = True):
    
    """
        For each single epoch, get dSPM results for freesurfer anatomical ROIs. 
        Should be able to apply to both MEG and EEG, by choosing the correct epoch_fname and fwd solutions 
        epoch_dir can point to both MEG and EEG data.
        Save a single variable per ROI. 
        Input:
            subj = "Subj1"
            epochs_fname, full path of the epochs file
            run_id = 1 to n_run
            fwd, forward object
            outdir, path to save the results
            labels, label_names,
            ROI_aggre_mode = "mean_flip" or "pca_flip"
        Output:
            Saved matdir 
            mat_dict = dict(ROI_ts = label_ts, ROI_names = label_names,
                        mode = ROI_aggre_mode, times = epochs.times,
                        epochs_fname = epochs_fname)
            Note: epochs_fname,  save it to indicate it was from MEG or EEG, and from what dataset            
    """
    MEGorEEG = ['EEG','MEG']
    epochs = mne.read_epochs(epochs_fname)                                    
    cov = mne.compute_covariance(epochs, tmin=None, tmax=-0.05)    
    inv_op = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, fixed = True)                                    
    snr = 1.0  # use lower SNR for single trial
    lambda2 = 1.0 / snr ** 2
    method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
    # set return_generator == True, so that the machine did not have to save the actual result
    # it is passed to the next stage for label ts
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2, method,
                                 return_generator=True)                            
    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator
    src = inv_op['src']
    label_ts = mne.extract_label_time_course(stcs, labels, src, mode= ROI_aggre_mode,
                                             return_generator = False)
    label_names = [label.name for label in labels]    
    n_trials = len(label_ts)         
    n_labels = len(labels)         
    n_times = len(epochs.times)                  
    label_ts_array = np.zeros([n_trials, n_labels, n_times])
    for i in range(len(label_ts)):
        label_ts_array[i] = label_ts[i]
    mat_dict = dict(ROI_ts = label_ts, ROI_names = label_names,
                    mode = ROI_aggre_mode, times = epochs.times,
                    epochs_fname = epochs_fname)
    mat_name = outdir + "%s_%s_%s_%s_run%d.mat" %(subj, MEGorEEG[isMEG], label_set_name, ROI_aggre_mode, run_id)
    scipy.io.savemat(mat_name, mat_dict, oned_as = "row")

#==============================================================================
def get_dSPM_ROI(subj, epochs_fname, run_id, fwd, outdir,
                             label, label_name, isMEG = True):
    
    """
    For each run, get dSPM for all source points in each ROI
    """
    MEGorEEG = ['EEG','MEG']
    epochs = mne.read_epochs(epochs_fname)                                    
    cov = mne.compute_covariance(epochs, tmin=None, tmax=-0.05)    
    inv_op = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, fixed = True)                                    
    snr = 1.0  # use lower SNR for single trial
    lambda2 = 1.0 / snr ** 2
    method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)
    # set return_generator == True, so that the machine did not have to save the actual result
    # it is passed to the next stage for label ts
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv_op, lambda2, method,
                                 label = label)                            
    # Average the source estimates within each label using sign-flips to reduce
    # signal cancellations, also here we return a generator
    ROI_ts = np.zeros( np.hstack([len(stcs), stcs[0].data.shape]) )
    for l in range(len(stcs)):
        ROI_ts[l] = stcs[l].data
    mat_dict = dict(ROI_ts = ROI_ts, ROI_names = [label_name],
                    mode = 0, times = epochs.times,
                    epochs_fname = epochs_fname)
    mat_name = outdir + "%s_%s_%s_run%d.mat" %(subj, MEGorEEG[isMEG], label_name, run_id)
    scipy.io.savemat(mat_name, mat_dict, oned_as = "row")
#==============================================================================

def match_dSPM_ROI_with_stim(subj, savedir, n_run, prefix, isMEG = True): 
    """
        Input:
            subj = "Subj1"
            epochs_fname, full path of the epochs file
            run_id = 1 to n_run
            fwd, forward object
            savedir = path where data is saved, i.e. outdir in  "get_dSPM_for_FS_anat_ROI"s
            label_set_name, name of the label_set
            SUBJECTS_DIR, environment variable for freesurfer/mne, Freesurfer folder
            ROI_aggre_mode = "mean_flip" or "pca_flip"
        Output (saved)
        mat_dict = dict(ROI_ts_each_trial = data_mat_no_repeat, im_id = im_id_no_repeat,
                    ROI_names = ROI_names, times = times,
                    epochs_fname = epochs_fname)
        Note: epochs_fname,  save it to indicate it was from MEG or EEG, and from what dataset   
        
    """
    data_list = list()
    if isMEG:
        for j in range(n_run):
            run_id = j+1
            mat_name = savedir + "%s_run%d.mat" %(prefix, run_id)
            mat_dict = scipy.io.loadmat(mat_name)
            print j, subj, run
            if subj in ['Subj9'] and run == "run4": 
                    # Note Subj 9 run 4 was interupted, it has to be handled differently, 
                    # only 168 trials were usable
                    tmp_n_trials = 168
                    data_list.append(mat_dict['ROI_ts'][0:tmp_n_trials])
            else:
                    data_list.append(mat_dict['ROI_ts'])
            if j == 0:
                ROI_names = mat_dict['ROI_names']
                times = mat_dict['times'][0]
                epochs_fname = mat_dict['epochs_fname']
                
        n_times = len(times) 
        n_ROI = len(ROI_names)   
        data_mat = data_list[0]
        for j in range(1,n_run):
            data_mat = np.vstack([data_mat, data_list[j]]) 
            
        #======================load the image sequences =================================================
        im_id = np.zeros(0)
        is_repeated = np.zeros(0)
        # for MEG for now, will add EEG later
        mat_file_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/Experiment_mat_files/"
        n_block, n_run_per_block = n_run//2,2
        for k in range(n_block):
            for l in range(n_run_per_block):
                if subj not in ['Subj1','Subj2','Subj3']:
                    tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_MEG_post_run.mat" %(k+1,l+1))
                else:
                    tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_post_run.mat" %(k+1,l+1))
                
                if subj in ['Subj9'] and  k == 1 and l == 1:
                    tmp_n_trial = 168 
                    im_id = np.hstack([im_id, tmp_mat['this_im_order'][0:tmp_n_trial,0],])
                    is_repeated = np.hstack([is_repeated, tmp_mat['this_is_repeated'][0:tmp_n_trial,0],])
                else:
                    # use the indice 0 to 361
                    im_id = np.hstack([im_id, tmp_mat['this_im_order'][:,0]])
                    is_repeated = np.hstack([is_repeated, tmp_mat['this_is_repeated'][:,0]])
    else:
        raise ValueError("currently for MEG only")
    im_id -= 1 # im_id should start from zero
    print "image id starts at %f" % im_id.min()
    data_mat_no_repeat = data_mat[is_repeated == 0]
    im_id_no_repeat = im_id[is_repeated == 0]    
    # save the first mat file
    mat_name = savedir + "%s_ROI_ts.mat" %(prefix)
    mat_dict = dict(ROI_ts_each_trial = data_mat_no_repeat, im_id = im_id_no_repeat,
                    ROI_names = ROI_names, times = times,
                    epochs_fname = epochs_fname)
    scipy.io.savemat(mat_name, mat_dict)

#==============================================================================
if __name__ == "__main__": 
    subj_list = range(1,14)
    n_run_per_subj_MEG = [6,12,6,12,10,8,12,10,10,10,12,12,12]
    n_subj = len(subj_list)
    isMEG = True
    ROI_aggre_mode = "mean_flip"
    # merge the left and right ROI
    label_set_name = "sceneROI"
    ROI_bihemi_names = [ 'medialorbitofrontal', 'pericalcarine', 
                        'PPA_c_g', 'TOS_c_g', 'RSC_c_g']
    labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"     
    for i in range(n_subj):
        subj= "Subj%d" %subj_list[i]
        labeldir1 = labeldir + "%s/" % subj
        # load and merge the labels
        labels  = list()
        labels_bihemi = list()
        ROI_names = list()
        for j in ROI_bihemi_names:
            tmp_label_list = list()
            for hemi in ['lh','rh']:
                print subj, j, hemi
                tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,j,hemi)
                print tmp_label_path
                tmp_label = mne.read_label(tmp_label_path)
                labels.append(tmp_label)
                ROI_names.append("%s-%s" %(j,hemi))
                tmp_label_list.append(tmp_label)
            labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1])  
        print labels
        # for the mean part, I can not obtain sign flipped mean for the joint left-right ROI
        # so run it on the ROIs on each hemisphere
        print ROI_names
        print labels_bihemi
        if isMEG: 
            fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                           + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
            fwd = mne.read_forward_solution(fwd_path, surf_ori = True)              
            for run in range(1,1+n_run_per_subj_MEG[i]):
                epochs_fname = ("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                                     +"MEG_DATA/DATA/epoch_raw_data/%s/%s_%s" \
                                    %(subj, subj, "run%d_filter_1_110Hz_notch_ica-epo.fif.gz" %run))
                outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"+\
                        "Result_MAT/source_solution/dSPM_MEG_ROI_single_trial/%s/" %subj 
                run_id = run
                if False:
                    get_dSPM_ROI_mean(subj, epochs_fname, run_id, fwd, outdir,
                                 labels, ROI_names, label_set_name, 
                                 ROI_aggre_mode = ROI_aggre_mode) 
                    print "ROI mean done"
                if False:
                    for l in range(len(labels_bihemi)):
                        print l
                        get_dSPM_ROI(subj, epochs_fname, run_id, fwd, outdir,
                                 labels_bihemi[l], ROI_bihemi_names[l], isMEG = True)
                    print "bihemi ROIs done"             
            savedir = outdir                     
            prefix = "%s_MEG_%s_%s" %(subj, label_set_name, ROI_aggre_mode)
            match_dSPM_ROI_with_stim(subj, savedir, n_run_per_subj_MEG[i], prefix, isMEG = True)
            
       
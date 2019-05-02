# -*- coding: utf-8 -*-
"""

alpha: threshold to drop bad trials.
Compute the average across repitions of images, where the number of reptitions in 
 each image is matched across MEG and EEG. 
After computing the averages, save them in two seperate files under their epoch directory. 
Also compute the empirical lag between the two modalities. 

"""

import numpy as np
import scipy.io
import scipy.stats


alpha = 15.0
n_im = 362
subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16, 18]
n_subj = len(subj_list)

meta_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"
MEG_dir = meta_dir + "MEG_DATA/DATA/epoch_raw_data/"
EEG_dir = meta_dir + "EEG_DATA/DATA/epoch_raw_data/"
save_name_suffix = "1_110Hz_notch_ica"

# for 
for EEG_save_name_suffix in [ "1_110Hz_notch_ica_PPO10POO10_swapped",
                         "1_110Hz_notch_ica"]:
    for i in range(n_subj):
        subj = "Subj%d" % subj_list[i]
        # load both data into a length 2 list, MEG first, EEG second
        mat_dict_list = list()
        MEG_fname = MEG_dir + "%s/%s_%s_all_trials.mat" %(subj, subj, save_name_suffix)
        EEG_fname = EEG_dir + "%s_EEG/%s_EEG_%s_all_trials.mat" %(subj, subj, EEG_save_name_suffix)
        mat_dict_list.append(scipy.io.loadmat(MEG_fname))
        mat_dict_list.append(scipy.io.loadmat(EEG_fname))
        
        # create a list with 362 elements corresponding to 362 images, 
        # each including the indices of the original data corresponding to repititions of the image
        ind_each_im_two_mod = np.zeros([2, n_im], dtype = np.object)  
        # count the valid repititions in each image, each modality, take the smallest in each case
        n_rep_valid_each_im_two_mod = np.zeros([2, n_im])
        # loop over the two modalities
        for l in range(2):
            n_trials = mat_dict_list[l]['epoch_mat_no_repeat'].shape[0]
            # each trial, each sensor, max across time - min across time
            ranges_each_trial = np.max(mat_dict_list[l]['epoch_mat_no_repeat'], axis = 2) \
                              - np.min(mat_dict_list[l]['epoch_mat_no_repeat'], axis = 2)
            ranges_zscore = scipy.stats.zscore(ranges_each_trial, axis = 0)
            bad_trials = np.any(ranges_zscore > alpha, axis = 1)
            print "# bad_trials %d" %bad_trials.sum()
            bad_trial_ind = np.nonzero(bad_trials)[0]
            # record the valid indices for each image
            for i in range(n_im):
                tmp_ind = np.nonzero(mat_dict_list[l]['im_id_no_repeat'][0]==i)[0]
                # remove the bad_trials 
                tmp_ind = np.setdiff1d(tmp_ind, bad_trial_ind)
                ind_each_im_two_mod[l,i] = np.array(tmp_ind).astype(np.int)
                n_rep_valid_each_im_two_mod[l,i] = len(tmp_ind)
            # debug 
            # something is wrong here
            #print n_rep_valid_each_im_two_mod[:,0:4]
                
        # compute the average, by taking first k repititions in each modality
        # where k the smallest number of valid repititions across the two modality
        out_name_list = [MEG_dir + "%s/%s_%s_MEEG_match_ave_alpha%1.1f.mat" \
                         %(subj, subj, save_name_suffix, alpha), \
                         EEG_dir + "%s_EEG/%s_EEG_%s_MEEG_match_ave_alpha%1.1f.mat" \
                         %(subj, subj, EEG_save_name_suffix, alpha)]
        for l in range(2):
            n_sensors, n_times = mat_dict_list[l]['epoch_mat_no_repeat'][0].shape
            ave_data = np.zeros([n_im, n_sensors, n_times])
            for i in range(n_im):
                if n_rep_valid_each_im_two_mod[l,i] == 0:
                    print "error! no trials found for l= %d, i = %d" %(l,i)
                    continue
                else:
                    tmp_k = np.int(np.min(n_rep_valid_each_im_two_mod[:,i]))
                    print "image %d common rep = %d" %(i, tmp_k)
                    tmp_ind = ind_each_im_two_mod[l,i][0:tmp_k]
                    ave_data[i] = mat_dict_list[l]['epoch_mat_no_repeat'][tmp_ind].mean(axis = 0)
            
            print "%d sensors found in modality %d" % (n_sensors, l)
            mat_dict_to_save = dict(ave_data = ave_data, times = mat_dict_list[l]['times'][0],
                            picks_all = mat_dict_list[l]['picks_all'][0],
                            n_rep_valid_each_im_two_mod = n_rep_valid_each_im_two_mod, 
                            alpha = alpha)
            scipy.io.savemat(out_name_list[l], mat_dict_to_save)
        
        del(mat_dict_list, out_name_list)
    #===============================================================
    # take an average and compute the MEEG lag? To be added or use the RDM    
    #================================================================
    # aslo remove the aspect ratio       
        
        
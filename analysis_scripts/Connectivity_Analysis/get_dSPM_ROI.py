import numpy as np
import scipy.io
import mne


def save_dSPM_ROI(ROI_labels, ROI_names, subj, isMEG, outdir, verbose = True):
    """
    Save the results in to seperate mat files for further use
    ROI_labels: list of labels
    ROI_names: names of the ROIs
    """
    #============== file names to load
    MEGorEEG = ['EEG','MEG']
    flag_swap_PPO10_POO10 = True
    MEG_fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"
    EEG_fname_suffix = "1_110Hz_notch_ica_PPO10POO10_swapped_ave_alpha15.0_no_aspect" \
        if flag_swap_PPO10_POO10 else "1_110Hz_notch_ica_ave_alpha15.0_no_aspect"  
    fname_suffix = MEG_fname_suffix if isMEG else EEG_fname_suffix
    
    stc_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/" \
                + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]
                
    if isMEG:
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                           + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
    else:
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                           + "EEG_DATA/DATA/fwd/%s_EEG/%s_EEG_oct-6-fwd.fif" %(subj, subj)
    
    #============== getting ROI indices
    fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
    src = fwd['src']
    
    ind0 = fwd['src'][0]['inuse']
    ind1 = fwd['src'][1]['inuse']
    # positions of dipoles
    rr = np.vstack([fwd['src'][0]['rr'][ind0==1,:], 
                             fwd['src'][1]['rr'][ind1==1,:]])
    rr = rr/np.max(np.sum(rr**2, axis = 1))                       
    nn = np.vstack([fwd['src'][0]['nn'][ind0 == 1,:],
                    fwd['src'][1]['nn'][ind1 == 1,:]])
        
    
    ROI_ind = list()
    for j in range(len(ROI_labels)):
        tmp_label = ROI_labels[j]
        _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
        ROI_ind.append(tmp_src_sel)
          
    #============== load and save source solutions       
    # load the source solution
    mat_dir = stc_out_dir + "%s_%s_%s_ave.mat" %(subj, MEGorEEG[isMEG],fname_suffix)
    mat_dict = scipy.io.loadmat(mat_dir)
    source_data = mat_dict['source_data']
    times = mat_dict['times'][0]
    del(mat_dict)
            
    for j in range(len(ROI_ind)):
        tmp = source_data[:,ROI_ind[j],:]
        tmp -= np.mean(tmp, axis = 0)  
        
        # also compute two different mean, "mean", "mean_sign_flip"
        mean = tmp.mean(axis = 1)
        
        tmp_nn = nn[ROI_ind[j],:]
        tmpu, tmpd, tmpv = np.linalg.svd(tmp_nn)
        tmp_sign = np.sign(np.dot(tmp_nn, tmpv[0]))
        #print tmp.shape
        #print tmp_sign.shape
        mean_sign_flip = np.mean(tmp.transpose([0,2,1])*tmp_sign, axis = -1)
        
        
        if verbose:
            print subj
            print ROI_names[j]
            print tmp.shape
            print mean.shape
            print mean_sign_flip.shape
     
        mat_dict = dict(dSPM_ROI = tmp, ROI_name = ROI_names[j], times = times,
                        mean = mean, mean_sign_flip = mean_sign_flip)     
        mat_name =  outdir  + "%s_dSPM_%s_%s.mat" %(subj,ROI_names[j], MEGorEEG[isMEG])
        scipy.io.savemat(mat_name, mat_dict)   
    
        
#===========================================       
if __name__ == "__main__":
    
    for isMEG in [False, True]:
        ROI_bihemi_names = ['pericalcarine', 'PPA_c_g', 'TOS_c_g', 'RSC_c_g', 'LOC_c_g',
                            'medialorbitofrontal'] 
        ROI_names = ['EVC','PPA','TOS','RSC','LOC','mOFC']     
              
        labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"  
        
        if isMEG:
            outdir ="/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_solution/dSPM_MEG_ave_per_im_ROI/"
        else:
            outdir ="/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/source_solution/dSPM_EEG_ave_per_im_ROI/"
    
        if isMEG:
            subj_list = range(1, 19)
        else:
            subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
            
        n_subj = len(subj_list)
        
        for i in range(n_subj): 
            subj = "Subj%d" %subj_list[i]
            labeldir1 = labeldir + "%s/labels/" % subj
            # load and merge the labels
            labels_bihemi = list()
            for j in range(len(ROI_names)):
                tmp_name = ROI_bihemi_names[j]
                tmp_label_list = list()
                for hemi in ['lh','rh']:
                    #print subj, tmp_name, hemi
                    tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,tmp_name,hemi)
                    tmp_label = mne.read_label(tmp_label_path)
                    tmp_label_list.append(tmp_label)
                labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1])
                
            save_dSPM_ROI(labels_bihemi, ROI_names, subj, isMEG, outdir)
 
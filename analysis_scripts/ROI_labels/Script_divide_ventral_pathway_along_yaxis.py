import numpy as np
import scipy.io 
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import mne


def divide_ventral_pathway_along_yaxis(subj, n_part = 4, quantile_flag = True):
    """
    # devide into n_part parts
    n_part = 4  
    # if quantile flag is true, divide using quantiles, else, equally devide y 
    quantile_flag = True  
    """
    # all ventral ROIs, except paricalcarine
    ROI_names = ['entorhinal', 'fusiform', 'inferiortemporal',
                 'lateraloccipital', 'lingual', 'parahippocampal', 'temporalpole']
    ROI_names_hemi = list()
    for hemi in ['lh','rh']:
        for j in range(len(ROI_names)):
            ROI_names_hemi.append(ROI_names[j] + "-%s" %hemi)       
                 
    SUBJECTS_DIR = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/" 
    fwd_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                                   + "MEG_DATA/DATA/fwd/"
    out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"

    # save everything in the format of src, not labels
    # load labels: merge the left and right ones
    fwd_path = fwd_dir + "%s/%s_ave-fwd.fif" %(subj, subj)
    fwd = mne.read_forward_solution(fwd_path, surf_ori = True)             
    ind0 = fwd['src'][0]['inuse']
    ind1 = fwd['src'][1]['inuse']
    # positions of dipoles
    rr = np.vstack([fwd['src'][0]['rr'][ind0==1,:], 
                             fwd['src'][1]['rr'][ind1==1,:]])
    rr = rr/np.max(np.sum(rr**2, axis = 1))                          
                 
    # load all the labels:
    labels = mne.read_labels_from_annot(subj, parc= "aparc",subjects_dir= SUBJECTS_DIR)
    # dipole indices
    dipole_ind = np.zeros(0, dtype = np.int) 
    label_subset = [label for label in labels if label.name in ROI_names_hemi]   
    for j in range(len(label_subset)):
        _, tmp_src_sel = mne.source_space.label_src_vertno_sel(label_subset[j], fwd['src']) 
        dipole_ind = np.hstack( [dipole_ind, tmp_src_sel])
        
    y_coord = rr[dipole_ind,1] 
    if quantile_flag:
        percentage = np.arange(0, 100+1E-2, 1.0/n_part*100.0) 
        # obtain the quantiles
        divide_seq = np.percentile(y_coord, percentage)  
    else: 
        ymin, ymax = y_coord.min(), y_coord.max()
        divide_seq = np.linspace(ymin, ymax, n_part+1)
    
    # 1:n_part
    dipole_label_val = np.zeros(rr.shape[0])
    for i in range(n_part): # [ ) intervals, except for the last one
        if i < n_part-1:
            tmp_ind = np.all(np.vstack( [y_coord>= divide_seq[i],  y_coord < divide_seq[i+1]]), axis = 0)
        else:
            tmp_ind = np.all(np.vstack( [y_coord>= divide_seq[i],  y_coord <= divide_seq[i+1]]), axis = 0)
        print "part %d %d dipoles" % (i+1, tmp_ind.sum())
        dipole_label_val[ dipole_ind[tmp_ind]] = i+1
     
    # save stc
    vertices = [ fwd['src'][0]['vertno'], fwd['src'][1]['vertno'] ]
    stc = mne.SourceEstimate(data = dipole_label_val[:,np.newaxis],
                             vertices = vertices, tmin = 0.0, tstep = 1.0, subject = subj)
    stc_name = out_dir + "%s/labels/%s_ventral_divide%d_label_quantile%d" %(subj, subj, n_part, quantile_flag)
    stc.save(stc_name)
    
    if False:
        # verified, dim1 is yaxis
        # visualize the results
        col_seq = ['m','r','y','g','b']
        plt.figure()
        plt.subplot(111, projection = '3d')
        plt.plot(rr[:,0],rr[:,1],rr[:,2], '.k')
        plt.plot(rr[dipole_ind,0],rr[dipole_ind,1],rr[dipole_ind,2], 'ok')
        
        plt.xlabel('dim0'); plt.ylabel('dim1');
        for i in range(n_part):
            tmp_ind = np.all( np.vstack([ dipole_label_val >= i+0.5, dipole_label_val < i+1.5]), axis = 0)
            _= plt.plot(rr[tmp_ind,0],rr[tmp_ind,1],rr[tmp_ind,2], "." + col_seq[i])
      

#================================================================================
subj_list = np.arange(1,19)
n_subj = len(subj_list)
for i0 in range(n_subj):
    subj = "Subj%d" %(subj_list[i0])
    for n_part in [3,4,5,6]:
        for quantile_flag in [False, True]:
            divide_ventral_pathway_along_yaxis(subj, n_part = n_part, quantile_flag = quantile_flag)
                        


    
    
                              



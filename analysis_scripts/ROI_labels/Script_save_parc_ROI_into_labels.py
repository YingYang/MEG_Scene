import numpy as np
import sys
import scipy.io
import scipy.stats
import mne

#================================================================================
def save_parc_ROI_into_labels(subj, outdir, parc = "aparc", SUBJECTS_DIR = None):
    """
      Save the parc 68 labels into label files
    """
    if SUBJECTS_DIR is None:
        SUBJECTS_DIR = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
        
    labels = mne.read_labels_from_annot(subj, parc= parc,
                                    subjects_dir= SUBJECTS_DIR) 
    for i in range(len(labels)):
        labels[i].save(outdir + "%s_%s.label" %(subj, labels[i].name) )
                              
#==============================================================================
   
if __name__ == "__main__":
    
    # save the ROIs into labels
    subj_list = range(1,19)
    n_subj = len(subj_list)
    parc = "aparc"
    outdir0 = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"
    for i in range(n_subj):
        subj = "Subj%d" % subj_list[i]
        outdir = outdir0 + "%s/labels/" % subj
        save_parc_ROI_into_labels(subj, outdir, parc = parc)

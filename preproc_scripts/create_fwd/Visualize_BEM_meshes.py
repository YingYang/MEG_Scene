import mne
import numpy as np
from mayavi import mlab

print(__doc__)

import scipy.stats

#data_path = sample.data_path()
#fname = data_path + '/subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif'


head_col = (0.9, 0.9, 0.9)  # light pink
#skull_col = (0.91, 0.89, 0.67)
#brain_col = (0.67, 0.89, 0.91)  # light blue
#colors = [head_col, skull_col, brain_col]
from mayavi import mlab  # noqa



subjects_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
epoch_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"
raw_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/raw_data/"

subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14,16,18]
n_subj = len(subj_list)


for i in range(n_subj):    
    subj = "Subj%d" %(subj_list[i])
    surface_fname = subjects_dir + subj + "/bem/" + subj+"-5120-5120-5120-bem-sol.fif"
    surfaces = mne.read_bem_surfaces(surface_fname, patch_stats=True)
    
                             
    
    # somehow mne was not able to read Subj14 epoch?              
    # load the digitization points
    #epoch_fname = "%s/Subj%d_EEG/Subj%d_EEG_filter_1_110Hz_notch_ica_reref-epo.fif.gz"\
    #       %(epoch_dir, subj_list[i], subj_list[i])
    #epochs = mne.read_epochs(epoch_fname)  
    
    raw_fname ="%s/Subj%d_EEG/Subj%d_EEG_raw.fif"\
           %(raw_dir, subj_list[i], subj_list[i])
    raw = mne.io.Raw(raw_fname)                
    
    dig = raw.info['dig']  # a list with 131 (128+3) elements, first three, fiducials
    coord = np.zeros([len(dig),3])
    for k in range(len(dig)):
        coord[k] = dig[k]['r']
        
    # Subject 11 has a terrible outlier, add one step to remove outliers
    z_score = scipy.stats.zscore(coord, axis = 0)
    valid = np.all(np.abs(z_score) < 10, axis = 1)
    coord = coord[valid,:]
       
    #trans_mat
    trans_fname = "/home/ying/dropbox_unsync/MEG_scene_neil"+\
     "/MEG_EEG_DATA/EEG_DATA/DATA/trans/Subj%d_EEG/Subj%d_EEG-trans.fif"\
     %(subj_list[i], subj_list[i])
    trans_dict = mne.read_trans(trans_fname)
    trans_mat = trans_dict['trans']
    coord_aug = np.hstack([coord, np.ones([coord.shape[0],1])])
    coord2 =  trans_mat.dot(coord_aug.T).T
    
    
    # Subject 11 has a terrible outlier
        
        
        
        
        
    mlab.figure(size=(600, 600), bgcolor=(0, 0, 0))
    #for c, surf in zip(colors, surfaces):
      
        
    mlab.points3d(coord2[:,0], coord2[:,1], coord2[:,2], 
                  opacity = 0.3, color = (1,0,0), scale_factor = 1E-2 )
                  
    c = head_col
    surf = surfaces[0]
    points = surf['rr']
    faces = surf['tris']
    mlab.triangular_mesh(points[:, 0], points[:, 1], points[:, 2], faces,
                         color=c, opacity=1.0)               
    
    
    outdir = "/home/ying/Dropbox/Scene_MEG_EEG/preproc_scripts/create_fwd/digi_scalp_alignment_eeg_fig/"
    view = ['front','back','left','right']
    
    a_seq = np.array([90,270,180,0])
    e_seq = np.array([90,90,90,90])
    for j in range(len(view)):
        mlab.view(azimuth=a_seq[j], elevation=e_seq[j], distance = 0.6)
        mlab.savefig( outdir+ "Subj%d_EEG_%s.png" %(subj_list[i], view[j]) )
    
    #mlab.points3d(coord[:,0], coord[:,1], coord[:,2], color = (1,0,0))



# read AFNI/SUMA 1d.roi files, save them into freesurfer label files. 
import numpy as np
import os
subj_list = np.arange(1,19)
n_subj = len(subj_list)

ROI_list = ['RSC_lh', 'RSC_rh','TOS_lh', 'TOS_rh','PPA_lh', 'PPA_rh', 
            'ObjFus_lh', 'ObjFus_rh', 'ObjLO_lh','ObjLO_rh']
roi_path = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"


ROI_list_new = ['LO1z_lh','LO1z_rh','LO2z_lh','LO2z_rh', 
                'PPAz_lh','PPAz_rh','RSCz_lh','RSCz_rh', 'TOSz_lh','TOSz_rh']
ROI_list_old = ['ObjLO_lh','ObjLO_rh','ObjFus_lh', 'ObjFus_rh',
                'PPA_lh','PPA_rh','RSCz_lh','RSCz_rh', 'TOSz_lh','TOSz_rh']




ROI_label_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"   


subj_list = range(1,19)
n_subj = len(subj_list)
n_ROI = len(ROI_list_new)

new_in_old_percent = np.zeros([n_subj, n_ROI])
Jaccard  = np.zeros([n_subj, n_ROI])

new_ROI_threshold = 0.1

           
for i in range(n_subj):
    subj = "Subj%d" % subj_list[i]
    for j in range(len(ROI_list)):
        print subj, ROI_list_new[j], ROI_list_old[j]
             
        tmp_data = list()
        roi1d_names = [ roi_path + "%s/suma_1D/%s_%s_SPM_v2s.1D.dset" %(subj, subj, ROI_list_old[j]),
                       roi_path + "%s/suma_1D/%s_%s_SPM_v2s_continuous.1D.dset" %(subj, subj, ROI_list_new[j]) ] 
        
        for l in range(2):
            tmp_data.append(np.loadtxt(roi1d_names[l]))
            
        ind1 = np.nonzero(tmp_data[1] >  new_ROI_threshold)[0]
        ind0 = np.nonzero(tmp_data[0])[0]
        
        common_ind = np.intersect1d(ind1, ind0)
        union_ind = np.union1d(ind1,ind0)
        ind1_diff_ind0 = np.setdiff1d(ind1, ind0)
            
        Jaccard[i,j] =np.float(common_ind.size)/np.float(union_ind.size)
        
        if ind1.size>0:
            new_in_old_percent[i,j] = 1.0- np.float(ind1_diff_ind0.size)/np.float(ind1.size)
        else:
            new_in_old_percent[i,j] = 1.0


import matplotlib.pyplot as plt      
plt.figure()
plt.subplot(1,2,1)
plt.imshow(Jaccard, vmin =0.0, vmax = 1.0, interpolation = "none")
plt.colorbar()  
plt.title('Jaccard')          
        
plt.subplot(1,2,2)
plt.imshow(new_in_old_percent, vmin = 0.0, vmax = 1.0, interpolation = "none")
plt.colorbar() 
plt.title('new in old percent')          
        
        
        
        
        
        
    

# read AFNI/SUMA 1d.roi files, save them into freesurfer label files. 
import numpy as np
import os
#subj_list = np.arange(1,14)
#Subj11 is not processed yet
#Subj10 does not have mFUS
subj_list = [1,2,3,4,5,6,7,8,9,10,12,13]
n_subj = len(subj_list)

ROI_list = ['LO_lh','LO_rh','mFUS_lh','mFUS_rh']
roi_path = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"
for i in range(n_subj):
    subj = "Subj%d" % subj_list[i]
    for j in range(len(ROI_list)):
        hemi = ROI_list[j][-2::]
        obj_contrast_name =  roi_path + "%s/suma_1D/%s_%s_%s_SPM_v2s.1D.dset" %(subj, subj,"spmT_0002",hemi )
        tmp_data = np.loadtxt(obj_contrast_name)
        #label files have indices starting at 0? Yes, it is 0-based!
        vertex_id = np.nonzero(tmp_data)[0]        
        roi_outer_name = roi_path + "%s/outer/%s_%s_outer.1D.roi" %(subj, subj, ROI_list[j])
        if os.path.isfile(roi_outer_name):
            print "outer bound found"
            tmp_outer = np.loadtxt(roi_outer_name)
            vertex_outer = tmp_outer[:,0].astype(np.int)
            vertex_id = np.intersect1d(vertex_id, vertex_outer)            

            n_vertex = len(vertex_id)
            hemi = ROI_list[j][-2::]
            # suffix _c standars corrected
            roi_out_fname = roi_path + "%s/%s_%s_c-%s.label" %(subj, subj, ROI_list[j][0:-3], hemi)
            fid = open(roi_out_fname, 'w')
            fid.write("#\n")
            fid.write("%d\n" % n_vertex)
            for k in range(n_vertex):
                fid.write("%d %1.6f %1.6f %1.6f %1.6f\n" %(vertex_id[k], 0,0,0,1))
            fid.close()
        
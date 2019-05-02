
# read AFNI/SUMA 1d.roi files, save them into freesurfer label files. 
import numpy as np
import os
subj_list = np.arange(1,19)
n_subj = len(subj_list)

ROI_list = ['RSC_lh', 'RSC_rh','TOS_lh', 'TOS_rh','PPA_lh', 'PPA_rh', 
            'ObjFus_lh', 'ObjFus_rh', 'ObjLO_lh','ObjLO_rh']
roi_path = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"



# in these cases, some parts of the first ROI were assigned to the second.
special_list = np.array([ ['Subj10', 'RSC_lh', 'PPA_lh'],
                 ['Subj12', 'ObjLO_rh','ObjFus_rh'],
                 ['Subj17', 'ObjLO_lh','ObjFus_lh'],
               ])

# Subj17 does not have left ObjFus_lh
no_ROI_list = np.array([ ['Subj17','ObjFus_lh']])               


for i in range(n_subj):
    subj = "Subj%d" % subj_list[i]
    for j in range(len(ROI_list)):
        print subj, ROI_list[j]
        if subj in no_ROI_list[:,0]:
            ind0 = np.nonzero(no_ROI_list[:,0] ==subj)[0][0]
            if ROI_list[j] == no_ROI_list[ind0,1]:
                print "%s %s does not exist"
                continue
               
        roi1d_name = roi_path + "%s/suma_1D/%s_%s_SPM_v2s.1D.dset" %(subj, subj, ROI_list[j])
        tmp_data = np.loadtxt(roi1d_name)
        #label files have indices starting at 0? Yes, it is 0-based!
        vertex_id = np.nonzero(tmp_data)[0]
        # Subj10, PPA also includes RSC regions
        if subj in special_list[:,0]:
            ind = np.nonzero(special_list[:,0] ==subj)[0][0]
            if ROI_list[j] == special_list[ind,1]:
                tmp_data1 = np.loadtxt(roi_path + "%s/suma_1D/%s_%s_SPM_v2s.1D.dset" %(subj, subj, special_list[ind,2]))
                vertex_id1 = np.nonzero(tmp_data1)[0]
                vertex_id = np.union1d(vertex_id1, vertex_id)
            else:
                pass
        
        roi_outer_name = roi_path + "%s/outer/%s_%s_outer.1D.roi" %(subj, subj, ROI_list[j])
        if os.path.isfile(roi_outer_name):
            print "outer bound found"
            tmp_outer = np.loadtxt(roi_outer_name)
            vertex_outer = tmp_outer[:,0].astype(np.int)
            vertex_id = np.intersect1d(vertex_id, vertex_outer)            

        n_vertex = len(vertex_id)
        hemi = ROI_list[j][-2::]
        # suffix _c standars corrected
        roi_out_fname = roi_path + "%s/labels/%s_%s_c-%s.label" %(subj, subj, ROI_list[j][0:-3], hemi)
        fid = open(roi_out_fname, 'w')
        fid.write("#\n")
        fid.write("%d\n" % n_vertex)
        for k in range(n_vertex):
            fid.write("%d %1.6f %1.6f %1.6f %1.6f\n" %(vertex_id[k], 0,0,0,1))
        fid.close()
        
import mne       
# merge the ObjLO and ObjFus?
LO_label_list = ['ObjLO', 'ObjFus']
for i in range(n_subj):
    subj = "Subj%d" % subj_list[i]
    for hemi in ['lh','rh']:
        label_list = list()
        for name in LO_label_list:
            label_name = roi_path + "%s/labels/%s_%s_c-%s.label" %(subj, subj, name, hemi)
            if os.path.isfile(label_name):
                label_list.append(mne.read_label(label_name))
            else:
                print "%s %s missing" %(subj, name+hemi)
        
        label0 = label_list[0]
        for i in range(1, len(label_list)):
            label0+= label_list[i]            
        # save label
        label0.save(roi_path + "%s/labels/%s_LOC_c-%s.label" %(subj, subj, hemi))
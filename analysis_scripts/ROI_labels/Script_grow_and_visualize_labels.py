import numpy as np
import sys
import scipy.io
import scipy.stats
import mne
import os


# save the ROIs into labels
subj_list = range(1,19)
n_subj = len(subj_list)
parc = "aparc"
#outdir0 = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"
#SUBJECTS_DIR = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"

uname = "ying"
data_dir0 = "/media/%s/Seagate Backup Plus Drive/" %uname+ \
        "CMU_work_2011_2017/psych-o-backup/MEG_NEIL/FREESURFER_ANAT/"

#outdir0 = "/home/yy/dropbox_unsync/MEG_scene_neil/ROI_labels/"
outdir0 = "/media/%s/Seagate Backup Plus Drive/" %uname + \
    "CMU_work_2011_2017/psych-o-backup/MEG_NEIL/ROI_labels/"
SUBJECTS_DIR = data_dir0

#%%
#============== check how many dipoles are in each ROI, if not enough, grow them=======
if False:
    n_dipole_min = 10
    # grow by 1 mm
    extents = 1.0
    ROI_names = ['PPA_c',
                 'TOS_c', 
                 'RSC_c', 
                 'LOC_c']
   
    ROI_n_dipole_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/ROI_n_dipole_table.txt"
    f = open(ROI_n_dipole_fname, 'w')
    f.write("Subj \t")
    for j in ROI_names:
        for hemi in ['lh','rh']:
            f.write("%s-%s \t" %(j, hemi))
    f.write("\n")
    for i in range(n_subj):
        subj = "Subj%d" % subj_list[i]
        outdir = outdir0 + "%s/labels/" % subj
        f.write("%d \t" %subj_list[i])
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                               + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
        fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
        src = fwd['src']
        for j in ROI_names:
            for hemi in ['lh','rh']:
                tmp_label_path  = outdir + "%s_%s-%s.label" %(subj,j,hemi)
                if os.path.isfile(tmp_label_path):
                    tmp_label = mne.read_label(tmp_label_path)
                    _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
                    tmp_n_dipole = len(tmp_src_sel)
                    f.write('%d \t' %tmp_n_dipole)
                    if True:
                        # only do it once
                        # expansion using grow_labels
                        #if the number of sources is smaller than a threshold, grow the label
                        hemi_ind = 0 if hemi is "lh" else 1
                        while tmp_n_dipole< n_dipole_min:
                            seeds = tmp_label.vertices
                            tmp_label_list = mne.label.grow_labels(subj,seeds,extents, hemi_ind, 
                                                              subjects_dir = SUBJECTS_DIR,
                                                              overlap = True)
                            # each element in the list grows into one ROI, now merge them
                            tmp_label = tmp_label_list[0]
                            for l in range(len(tmp_label_list)-1):
                                tmp_label+= tmp_label_list[l+1]                                  
                            _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
                            tmp_n_dipole = len(tmp_src_sel)
                            print "%s %s %d" %(j, hemi, tmp_n_dipole)
                        # g stands for grow    
                        print "saving grown label"
                        tmp_label.save(outdir+"%s_%s_g-%s.label" %(subj,j,hemi))  
                else:
                    f.write('%d \t' %0)                                                    
        f.write("\n")
    f.close()
      

    # for each subject, if the LOC overlapps with any of the scene ROIs, exclude them
    ROI_list0 = ['LOC']
    ROI_list1 = ['PPA','RSC','TOS']
    for i in range(n_subj):
        subj = "Subj%d" % subj_list[i]
        outdir = outdir0 + "%s/labels/" % subj
        for hemi in ['lh','rh']:
            label_list = list()
            for name in ROI_list1:
                label_list.append(mne.read_label(outdir+"%s_%s_c_g-%s.label" %(subj, name, hemi)))
            
            for name0 in ROI_list0:
                tmp_label = mne.read_label(outdir+"%s_%s_c_g-%s.label" %(subj, name0, hemi))
                for j in range(len(label_list)):
                    tmp_label = tmp_label-label_list[j]
                
                tmp_label.save(outdir+"%s_%s_c_g-%s.label" %(subj, name0, hemi))
            

# print the new number of dipoles in each ROI
if False:  
    ROI_names = ['PPA', 'TOS', 'RSC', 'LOC']
    ROI_n_dipole_fname = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/ROI_n_dipole_table_after_correction.txt"
    f = open(ROI_n_dipole_fname, 'w')
    f.write("Subj \t")
    for j in ROI_names:
        for hemi in ['lh','rh']:
            f.write("%s-%s \t" %(j, hemi))
    f.write("\n")
    for i in range(n_subj):
        subj = "Subj%d" % subj_list[i]
        outdir = outdir0 + "%s/labels/" % subj
        f.write("%d \t" %subj_list[i])
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                               + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
        fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
        src = fwd['src']
        for j in ROI_names:
            for hemi in ['lh','rh']:
                tmp_label_path  = outdir + "%s_%s_c_g-%s.label" %(subj,j,hemi)
                if os.path.isfile(tmp_label_path):
                    tmp_label = mne.read_label(tmp_label_path)
                    _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
                    tmp_n_dipole = len(tmp_src_sel)
                    f.write('%d \t' %tmp_n_dipole)
                else:
                    print "file not found"
                    f.write('%d \t' %0)                                                    
        f.write("\n")
    f.close()


#============== render a visualization of the ROIs ============================
# pysurfer supports directly ploting the labels with customized color
#https://pysurfer.github.io/examples/plot_label.html
#%%
if True:  
    
    '''
    20180730
    make sure to run 
    export ETS_TOOLKIT=qt 
    before running python
    run the script through command line
    
    Tried everything, there was not a platform that I could re-plot this. 
    Mayavi issue, suma issues, opengl issues. .....
    
    I am going to keep the original plot as supplementary plot in the paper. 
    
    '''
    from surfer import Brain
    os.environ["SUBJECTS_DIR"] = SUBJECTS_DIR
    
    #ROI_list = ['PPA_c_g','TOS_c_g','RSC_c_g','LOC_c_g', 'pericalcarine', 'medialorbitofrontal']
    ROI_list = ['PPA_c_g','TOS_c_g','RSC_c_g','LOC_c_g', 'pericalcarine']
    # color is not working
    # debug
    #n_subj = 1
    #ROI_list = ['PPA_c_g', 'TOS_c_g']
    
    #col = ['red','yellow','magenta','blue', 'green','cyan']
    col = ['r','y','m','b', 'g','c']
    col = [np.array([0,0,1]),
           np.array([1,1,0])]
    figdir = "/home/%s/Dropbox/Scene_MEG_EEG/ROI_visualization/" %uname
    
    surf = "inflated"
    hemi_seq = ["lh","rh"]
    view_list = [
    [# left ventral
    dict(azimuth=0, elevation=180, distance=150, focalpoint=[0, -10, 0]),
    # left medial
    dict(azimuth=-20, elevation=100, distance=125, focalpoint=[0, -5, 0]),
    # left occiptial
    dict(azimuth=50, elevation=-100, distance=100, focalpoint=[0, -20, 0])],
    [# right ventral
    dict(azimuth=0, elevation=180, distance=150, focalpoint=[0, -10, 0]),
    # right medial
    dict(azimuth=-20, elevation=-100, distance=125, focalpoint=[0, -5, 0]),
    # right occiptial
    dict(azimuth=130, elevation=-100, distance=100, focalpoint=[0, -20, 0])]
    ]
    view_name = ["ventral","medial","occ"]
    for i in range(n_subj):
        subject_id = "Subj%d" %(i+1)
        surf = "inflated"
        for j in range(len(hemi_seq)):
            hemi = hemi_seq[j]
            brain = Brain(subject_id, hemi, surf, background = "white",
                          subjects_dir = SUBJECTS_DIR)
            for l in range(len(ROI_list)):
                name = ROI_list[l]
                tmp_label_name = outdir0 + "%s/labels/%s_%s-%s.label" \
                                    %(subject_id, subject_id,name, hemi)
                #brain.add_label(tmp_label_name, color = col[l], alpha = 0.7)
                brain.add_label(tmp_label_name, color="#F0F8FF", alpha = 0.3)
                print ("color")
                print (l, col[l])
            
        
            for l in range(len(view_name)):
                brain.show_view(view_list[j][l])
                brain.save_image(figdir+"%s_%s_%s_SceneObj_ROIs.pdf"
                               %(subject_id, view_name[l], hemi))                    
            
            brain.close()

# debug 
# ventral
#brain.show_view(dict(azimuth=0, elevation=180, distance=200, focalpoint=[0, -10, 0]))
# medial
#brain.show_view(dict(azimuth=20, elevation=-100, distance=125, focalpoint=[0, -5, 0]))
# right occiptial
#brain.show_view(dict(azimuth=110, elevation=-100, distance=0, focalpoint=[0, -20, 0]))
      
        
        
#%%
# Debug color issue
'''
import os
from surfer import Brain

print(__doc__)

subject_id = "fsaverage"
hemi = "lh"
surf = "smoothwm"
brain = Brain(subject_id, hemi, surf)

# If the label lives in the normal place in the subjects directory,
# you can plot it by just using the name
brain.add_label("BA1_exvivo")

# Some labels have an associated scalar value at each ID in the label.
# For example, they may be probabilistically defined. You can threshold
# what vertices show up in the label using this scalar data
brain.add_label("BA1_exvivo", color="blue", scalar_thresh=.5)

# Or you can give a path to a label in an arbitrary location
subj_dir = os.environ["SUBJECTS_DIR"]
label_file = os.path.join(subj_dir, subject_id,
                          "label", "%s.MT_exvivo.label" % hemi)
brain.add_label(label_file)

# By default the label is 'filled-in', but you can
# plot just the label boundaries
brain.add_label("BA44_exvivo", borders=True)

# You can also control the opacity of the label color
brain.add_label("BA6_exvivo", alpha=.7)

# Finally, you can plot the label in any color you want.
brain.show_view(dict(azimuth=-42, elevation=105, distance=225,
                     focalpoint=[-30, -20, 15]))

# Use any valid matplotlib color.
brain.add_label("V1_exvivo", color="steelblue", alpha=.6)
brain.add_label("V2_exvivo", color="#FF6347", alpha=.6)
brain.add_label("entorhinal_exvivo", color=(.2, 1, .5), alpha=.6)
'''
#%%
'''
20180730 
newly added by Ying. 
Because of mayavi qt issues, I was not able to redraw the ROIs. 
The work around is to use SUMA and plot individual ROIs for one example subject
So I had to convert things into 1D. 
These are added to the Seagate Back UP (cluster copy)

The tarrlab hard disk copy of FREESURFER_ANAT 
seems to be damaged as SUMA was not able to open them. 

Unfortunately SUMA was not able to display it
'''
if False:
    # save the ROIs in to 1D files
    #ROI_list = ['PPA_c_g','TOS_c_g','RSC_c_g','LOC_c_g', 'pericalcarine', 'medialorbitofrontal']
    ROI_list = ['PPA_c_g','TOS_c_g','RSC_c_g','LOC_c_g', 'pericalcarine']
    
    ROI_integer = [1,2,3,4,5]
    hemi_seq = ["lh","rh"]
    col = ['red','yellow','magenta','blue', 'green','cyan']
    #figdir = "/home/yy/Dropbox/Scene_MEG_EEG/ROI_visualization/"
    
    for i in range(n_subj):
        subject_id = "Subj%d" %(i+1)
        for j in range(len(hemi_seq)):
            hemi = hemi_seq[j]
            for l in range(len(ROI_list)):
                name = ROI_list[l]
                tmp_label_name = outdir0 + "%s/labels/%s_%s-%s.label" \
                                    %(subject_id, subject_id,name, hemi)
                                    
                with open(tmp_label_name) as fid:
                    tmp_data = fid.readlines()
                    #tmp_data = tmp_data[1::]
                    
                    tmp_out_label_name = outdir0 + "%s/labels/%s_%s-%s.1D" \
                                    %(subject_id, subject_id,name, hemi)
                    with open(tmp_out_label_name, 'w') as fid2:
                        for l1 in range(2, len(tmp_data)):
                            fid2.write("%s %d\n" % ( tmp_data[l1].split()[0],
                                                    ROI_integer[l]))
                    
                    

              
          
          
          
          
          
# ====================== merge the left and right ROIs =======================
# Not working, BiHemiLabels could not be saved at all
#if False:
#    ROI_names = [ 'medialorbitofrontal', 'pericalcarine', 'lateraloccipital','PPA_c', 'TOS_c', 'RSC_c']
#    for i in range(n_subj):
#        subj = "Subj%d" % subj_list[i]
#        outdir = outdir0 + "%s/" % subj
#        for j in ROI_names:
#            tmp_label_list = list()
#            for hemi in ['lh','rh']:
#                tmp_label_path  = outdir + "%s_%s-%s.label" %(subj,j,hemi)
#                print tmp_label_path
#                tmp_label_list.append(mne.read_label(tmp_label_path))
#            print tmp_label_list
#            tmp_label = tmp_label_list[0] + tmp_label_list[1]
#            print tmp_label
#            # BiHemiLabel type object can not be saved
#            #tmp_label.save(outdir+"%s_%s-both.label" %(subj,j))
#!/bin/tcsh

# For MEG
# after obtaining the trans files, compute the forward solutions for each run.
setenv SUBJECTS_DIR "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
set transdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/trans"
set fwddir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/fwd"
set rawdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/raw_data"
set freesurferdir = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT"
# subject 1 to 9, to be updated
set n_runs_per_subj = (6 12 6 12 10 8 12 10 10 10 12 12 12 12 10 12 12 12)
#foreach i (`seq 1 1 8`) # first 8 subjects
set src = "oct-6"
set src = "ico-4"
#foreach i (`seq 1 1 13`) # subjects 9 to 13
foreach i (`seq 14 1 18`)
    set subj = "Subj$i"
    # each subject has different number of runs
    set n_run = ${n_runs_per_subj[$i]}
    foreach run (`seq 1 1 ${n_run}`)
        mne_forward_solution \
        --src $SUBJECTS_DIR/$subj/bem/${subj}-${src}-src.fif \
        --bem $SUBJECTS_DIR/$subj/bem/${subj}-5120-5120-5120-bem-sol.fif \
        --mri $transdir/$subj/${subj}-trans.fif \
        --meas $rawdir/$subj/intact/NEIL_${subj}_run${run}_raw.fif \
        --fwd $fwddir/$subj/${subj}_${src}_run${run}-fwd.fif \
        --meg --mindist 2.5 --accurate
    end
end

# note: Subj1-13: Subj*_run*-fwd.fif
# Subj*_ave-fwd.fif are computed from the src oct-6. 

# mindist should be the size of the BEM triangles in mm
# we have 5120 triangles, if human head radius is around 75 mm, the surface area of half a sphere is 2pi r**2 
# each triangle has a area of 2pi*r**2/5120 = 6.89 mm,  so the size of the triangle = np.sqrt(6.89*2) = 3.7mm


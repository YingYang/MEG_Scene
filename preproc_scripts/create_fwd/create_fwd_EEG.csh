#!/bin/tcsh

# For MEG
# after obtaining the trans files, compute the forward solutions for each run.
setenv SUBJECTS_DIR "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
set transdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/trans"
set fwddir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/fwd"
set rawdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/raw_data"
set freesurferdir = "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT"


foreach src ("oct-6" "ico-4")
#set src = "oct-6"
#set src = "ico-4"
# I have only obtained okay trans file for Subj4
	foreach i (1 2 3 4 5 6 7 8 10 11 12 13 14 16 18)
	    set subj = "Subj$i"
	    # each subject has different number of runs
		mne_forward_solution \
		--src $SUBJECTS_DIR/$subj/bem/${subj}-${src}-src.fif \
		--bem $SUBJECTS_DIR/$subj/bem/${subj}-5120-5120-5120-bem-sol.fif \
		--mri $transdir/${subj}_EEG/${subj}_EEG-trans.fif \
		--meas $rawdir/${subj}_EEG/${subj}_EEG_raw.fif \
		--fwd $fwddir/${subj}_EEG/${subj}_EEG_${src}-fwd.fif \
		--eeg --mindist 2.5 --accurate
	end
end

# mindist should be the size of the BEM triangles in mm
# we have 5120 triangles, if human head radius is around 75 mm, the surface area of half a sphere is 2pi r**2 
# each triangle has a area of 2pi*r**2/5120 = 6.89 mm,  so the size of the triangle = np.sqrt(6.89*2) = 3.7mm


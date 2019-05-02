#!/bin/tcsh

setenv SUBJECTS_DIR "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"
#====================== Subj 1-8 ===================================
foreach i (`seq 1 1 8`)
        set subj = "Subj$i"
	# create the source space
	# ico-6, 4098 source points per hemishpere, 4.9 mm, 24 mm2 per source
	mne_setup_source_space --subject $subj --ico -6 --overwrite
        # ico-5, 1026 source points per hemisphere, 9.9 mm, 97 mm2 per source. 
        mne_setup_source_space --subject $subj --ico -5 --overwrite
end

#====================== Subj 9-13 ====================================
foreach i (`seq 9 1 13`)
        set subj = "Subj$i"
	# create the source space
	# ico-6, 4098 source points per hemishpere, 4.9 mm, 24 mm2 per source
	mne_setup_source_space --subject $subj --ico -6 --overwrite
        # ico-5, 1026 source points per hemisphere, 9.9 mm, 97 mm2 per source. 
        mne_setup_source_space --subject $subj --ico -5 --overwrite
end

#===================== Subj 14-18 ===========================
foreach i (`seq 14 1 18`)
        set subj = "Subj$i"
	# create the source space
	# ico-6, 4098 source points per hemishpere, 4.9 mm, 24 mm2 per source
	mne_setup_source_space --subject $subj --ico -6 --overwrite
        # ico-5, 1026 source points per hemisphere, 9.9 mm, 97 mm2 per source. 
        mne_setup_source_space --subject $subj --ico -5 --overwrite
end



#======================================================================


# added for nips simulations, but could be useful later
#=========== additionaly, add --ico 4, 2562 sources per hemi, 6.2mm, 39mm2 per source
foreach i (`seq 1 1 13`)
        set subj = "Subj$i"
        mne_setup_source_space --subject $subj --ico 4
end

#===========================================================
foreach i (`seq 14 1 18`)
        set subj = "Subj$i"
        mne_setup_source_space --subject $subj --ico 4
end






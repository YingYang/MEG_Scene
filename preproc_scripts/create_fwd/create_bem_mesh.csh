#!/bin/tcsh

setenv SUBJECTS_DIR "/home/ying/dropbox_unsync/MEG_scene_neil/FREESURFER_ANAT/"

#================================ Subj 1-8 =================================
foreach i (1 3 4 5 7 6 7) # weird, should be 1 3 4 5 6 7
        # setting up the boundary element surface
	# ico 4 was recommanded 
	set subj = "Subj$i"
	mne_setup_forward_model --subject $subj --surf --ico 4
        # out puts Subj1-outer_skin-5120.surf, Subj1-outer_skull-5120.surf, etc
end

# for these two Subj 2 8 the inner and outer skull had intersection, try move outer skull outward
# only move the minimum so that it can run 
#--outershift shift [mm] Shift outer skull outwards by this amount (mm)
mne_setup_forward_model --subject Subj2 --surf --ico 4 --outershift 0.035 # mm
mne_setup_forward_model --subject Subj8 --surf --ico 4 --outershift 1.095

#================================= Subj 9 - 13 ============================
foreach i (9 10 11 12 13) 
        # setting up the boundary element surface
	# ico 4 was recommanded 
	set subj = "Subj$i"
	mne_setup_forward_model --subject $subj --surf --ico 4
        # out puts Subj1-outer_skin-5120.surf, Subj1-outer_skull-5120.surf, etc
end

#================================= Subj 14 - 18
foreach i (14 15 16 17 18) 
        # setting up the boundary element surface
	# ico 4 was recommanded 
	set subj = "Subj$i"
	mne_setup_forward_model --subject $subj --surf --ico 4
        # out puts Subj1-outer_skin-5120.surf, Subj1-outer_skull-5120.surf, etc
end



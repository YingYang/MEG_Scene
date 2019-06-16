# MEG_Scene
code for preprocessing and analysis for the paper "Exploring spatio-temporal neural dynamics of the human visual cortex" 
(accepted by Human Brain Mapping DOI to be added,
preprint available at https://www.biorxiv.org/content/biorxiv/early/2018/09/20/422576.full.pdf)

## Code
TBA


## Dataset

The preprocessed MEG data from 18 participants are available at Figshare (figshare.com/articles/MEG_scene_data/7991615. 
These data have the following file names and contents. 
- `Subj*_1_110Hz_notch_ica_all_trials.mat`, which contains all trials (including repetitions of the images). 
- `Subj*_1_110Hz_notch_ica_ave_alpha15.0.mat`, which contains the averaged neural responses to each stimulus image. 
   The data contains an `[n_images x n_channels x n_times]` matrix in mat file format, 
   where the order of the images matches the image id in `STIM.zip', 
   the suffix `15` indicates how the outlier was removed (if in a trial, for any channel 
   the difference betwen the max and min in the trial time was greater than mean + 15* standard deviation, 
   then the trial was removed from computing the average). 
- `Subj*_1_110Hz_notch_ica_ave_alpha15.0_noaspect.mat`, also averaged neural responses to the 362 images,
 where the correlation with the nuisance factors related to image size and aspect ratio were removed. 
 See the pre-print for more details. 

The dataset in this Figure share link also contains the following:
- `STIM.ZIP` which contains the following folders: 
 ** `Extra_Images`: 1086 extra images that were used in generating some features. 
 ** `Images`: 362 images that were presented to the participants
 ** `PTB_data/MEG`: the log of the image presentation sequence
 ** `Presentation_script_20151002.zip`: the PsychToolBox presentation scripts in Matlab. 
- `ROI_labels.zip` that includes `Subj*.zip`, in which the `labels` folder contains ROI `*.label` files 
  that marked the source points in the regions of interest. 
- `fwd.zip` contains the forward solutions for each participants that can project the sensor space data to source space. 
- `MEG_NEIL_Image_SUN_hierarchy_manual_20160617.csv` include the scene categories of the 362 images and some manual annotation about their semantic categories. 


Please download the features from the following dropbox link (TBA). 

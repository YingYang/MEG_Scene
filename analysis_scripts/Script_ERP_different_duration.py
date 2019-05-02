import mne
import matplotlib.pyplot as plt
import numpy as np

epoch_suffix = "filter_1_110Hz_notch_ica_reref"
EEG_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/epoch_raw_data/"

# load all subjects epochs, and plot the one channel's evoked response

Subj_list = ['SubjYY_100','SubjYY_500', 
             'Subj1','Subj2','Subj3', 'Subj4','Subj5',
             'Subj6','Subj7','Subj8','Subj10','Subj12', 
             'Subj13','Subj14','Subj16','Subj18']

col_seq  = ['r','g',
           'b','b','b','b','b',
           'b','b','b','b','b',
           'b','b','b','b']

posterior_ind = range(120,127) + range(54,64)



posterior_data = np.zeros([len(Subj_list), 564])

for i in range(len(Subj_list)):
    subj = Subj_list[i]
    tmp_epoch_name = "%s/%s_EEG/%s_EEG_%s-epo.fif.gz" %(EEG_dir, subj, subj, epoch_suffix)
    epochs = mne.read_epochs(tmp_epoch_name)
    evoked = epochs.average()
    tmp_data = evoked.data[posterior_ind,:].mean(axis = 0)
    times = evoked.times
    posterior_data[i] = tmp_data


plt.figure()
for i in range(len(Subj_list)):    
    plt.plot(times, posterior_data[i], col_seq[i], alpha = 0.5)
plt.legend(Subj_list)
plt.grid('on')
    
    

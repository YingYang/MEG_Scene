# -*- coding: utf-8 -*-
"""
Subj1 run1 was too long, Maxfilter can not handle it. 
The first 100s does not have any stimulus, crop it. 
"""


import mne
mne.set_log_level('WARNING')


tmp_root_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/DATA/raw_data/"

subj = "Subj1"
    
# load the channels to remove
raw_file_dir = tmp_root_dir  + subj + "/"
raw_name = raw_file_dir + "intact/NEIL_%s_run1_raw.fif" %(subj)
raw = mne.io.Raw(raw_name)
raw.info['bads'] =[]
       
# change the line frequency too
raw.info['line_freq'] = 60.0
print raw.info['line_freq']
# same the same file, overwrite
raw_name1 = raw_file_dir + "intact/NEIL_%s_run1_crop_raw.fif" %(subj)
events = mne.find_events(raw,stim_channel = "STI101",min_duration = 2/raw.info['sfreq'],
                         verbose = True)

# weird, events time does not match the true time?
# the difference is 78s
# but the epoched data is fine, 
epochs = mne.Epochs(raw, events, event_id = 1, tmin = 0.0, tmax = 0.5, proj = False)



if False:
    data1,_ = raw[0,events[1,0]-78000:events[1,0]+500-78000]
    
    plt.figure()
    #plt.subplot(1,2,1)
    plt.plot(data1.T)
    #plt.subplot(1,2,2)
    plt.plot(epochs[0].get_data()[0,0,:])
    
    plt.plot(epochs[0].get_data()[0,0,0:500], data1[0], '.')
    
    import matplotlib.pyplot as plt
    data, times = raw[-4,0::]
    plt.figure()
    plt.plot(times,data[0])

raw1 = raw.crop(tmin = 136.0, tmax = None)
raw1.save(raw_name1, overwrite = True)
        
   
    




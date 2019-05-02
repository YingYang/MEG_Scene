import mne
import numpy as np
import scipy
import scipy.io

tmp_rootdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/DATA/"
raw_dir = tmp_rootdir + "raw_data/"
filtered_dir = tmp_rootdir + "filtered_raw_data/"
ica_dir = tmp_rootdir + "ica_raw_data/" 
l_freq = 1.0
h_freq = 110.0
notch_freq = [60.0,120.0]

fname_suffix = "filter_%d_%dHz_notch_raw" %(l_freq, h_freq)
Masteroids = ['M1','M2']
#EOG_list = [u'LhEOG', u'RhEOG', u'LvEOG1', u'LvEOG2', u'RvEOG1']
EOG_list = ['EOG_LO1','EOG_LO2','EOG_IO1','EOG_SO1','EOG_IO2']
#ECG_list = [u'ECG']
ECG_list = ['ECG']

drop_names = []
for i in range(7):
    drop_names.append("misc%d"%(i+1))

trigger_list = ['STI101']      
# the trigger is now status 
exclude_list = Masteroids + EOG_list + ECG_list + drop_names + trigger_list

# subject
#============================================
#subj_id_seq = [1,2,3,4,5,6,7,8,10,11,12,13]    
#subj_list = ['Extra1','Extra2'] 
#for i in subj_id_seq:
#    subj_list.append('Subj%d' % i)
#subj_list = ['Subj14']  
#subj_list = ['Subj16', 'Subj18']  
#subj_list = ['SubjYY_100', 'SubjYY_500']

# additional EEG subjects 500 ms presentation
subj_list = ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]

      
n_subj = len(subj_list) 

#============================================    
for i in range(n_subj):
    subj = subj_list[i]
    print subj
    raw_fname = raw_dir + "%s_EEG/%s_EEG_raw.fif" %(subj, subj)
    raw = mne.io.Raw(raw_fname, preload = True, proj = False)
    raw.info['projs'] = []
    
    # 
    # bad_channels were identified in Step1
    #bad_channel_list_name = raw_dir + "%s_EEG/%s_EEG_bad_channel_list.txt" %(subj,subj)
    # ignore comments #, the first line is always the emtpy room data
    #bad_channel_list = list()
    #f = open(bad_channel_list_name)
    #for line in f:
    #    if line[0]!= "#":
    #        print line.split()
    #        bad_channel_list.append(line.split())
    #f.close()
    # there will be only one line
    #raw.info['bads'] += bad_channel_list[0]
    #print "bad channels"
    #print raw.info['bads']


    # cut the ending part which may mess everything up
    # 20161222: additional subjects, presentation = 500 ms, original code did not work
    # minimum duration of events should be larger 1.0/raw.info['sfreq']
    if subj in ["Subj_additional_1_500", "Subj_additional_2_500", "Subj_additional_3_500"]:
        events = mne.find_events(raw, stim_channel = trigger_list[0], min_duration =  1.1/raw.info['sfreq'])
    else:        
        events = mne.find_events(raw, stim_channel = trigger_list[0] )
    start_time = max(raw._times[events[0,0]]-5.0,0.0)
    end_time = raw._times[events[-1,0]] + 5.0
    raw = raw.crop(tmin = start_time, tmax = end_time, copy = True)
    # the Status channel has extremely large values
    # subtract the minium value there, so it is 0, 100, 200
    
    # SubjYY_500, after 922s, the referance failed, everything became periodical pulse
    if subj is "SubjYY_500":
        raw = raw.crop(tmin = 0.0, tmax = 922.0)
    
    
    # modified 20160530
    # for Subj14, min of STI101 is not correct
    if subj is "Subj14":
        print "use frequent counts"
        tmp = raw._data[-1,:].copy()
        counts = np.bincount(tmp[0::100].astype(int))
        baseline = np.argmax(counts)
        raw._data[-1,:] -= baseline
    else:
        raw._data[-1,:] -= np.min(raw._data[-1,:])
    picks = mne.pick_channels(raw.info['ch_names'],include = raw.info['ch_names'],
                            exclude = drop_names + trigger_list+raw.info['bads'])                       
    # ================remove the DC for each individual channel?================
    data = raw._data[picks,:]
    data = (data.T- np.mean(data, axis = 1)).T
    raw._data[picks,:] = data
    
    #=================== do not reference to mastoids M1, M2==========================
    # if M1/M2 are very noisy, it will mess up all channels!!!
    # raw, ref_data = mne.io.set_eeg_reference(raw, ref_channels=Masteroids, copy=True)
    # also remove the irrelevent channels, they should all be zero. 
    drop_picks = mne.pick_channels(raw.info['ch_names'],include =drop_names)
    raw._data[drop_picks] = 0
        
    raw.filter(l_freq, h_freq, picks = picks)
    raw.notch_filter(notch_freq)
    filtered_name = filtered_dir + "%s_EEG/%s_EEG_%s.fif" %(subj, subj, fname_suffix)
    raw.save(filtered_name, overwrite = True)
    
    #raw.plot()

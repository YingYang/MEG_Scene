import mne
import numpy as np
import scipy.io
import glob

# question by Reviewer 2
# load one of the raw data, after ica, compute the variance explained by the proj (SSP)


data_root_dir0 = "/media/yy/LinuxData/yy_Scene_MEG_data/"
data_root_dir =data_root_dir0 +"MEG_preprocessed_data/"


on_disk_data_dir = "/media/yy/Seagate Backup Plus Drive/" + \
    "CMU_work_2011_2017_seagate/psych-o-backup/MEG_NEIL/MEG_RAW_DATA/"


l_freq = 1.0
h_freq = 110.0
notch_freq = [60.0,120.0]



trace_ratio_all = []

for i in range(18):
    
    subj = "Subj%d" %(i+1)
    reject_dict = dict(grad=2000e-13, # T / m (gradiometers)
                  mag=3e-12, # T (magnetometers)
                  )
    
        
    empty_room_fname = glob.glob("%s/%s/??????/*%s*empty*_raw.fif" 
                           %( on_disk_data_dir, subj,subj))[0]
    
    data_fnames = np.sort(glob.glob("%s/%s/??????/*%s*run*_raw.fif" 
                             %( on_disk_data_dir, subj,subj)))
    print (data_fnames)
    
    
    
    er_raw = mne.io.Raw(empty_room_fname, preload = True)
    er_raw.notch_filter(notch_freq, n_jobs = 2)
    er_raw.filter(l_freq, h_freq, n_jobs = 2) 
    er_raw.add_proj([], remove_existing = True)
    projs = mne.proj.compute_proj_raw(er_raw, duration = 2.0, n_grad = 2, n_mag = 3,
                                              n_jobs = 2, reject = reject_dict)
    
    
    trace_ratio = np.zeros(0)
    for l, fname in enumerate(data_fnames):
        tmp_raw = mne.io.Raw(fname)
        raw = mne.io.Raw(fname, preload = True)
        raw.filter(l_freq, h_freq, n_jobs = 2)
        raw.notch_filter(notch_freq, n_jobs = 2)
        
        cov0 = mne.compute_raw_covariance(raw)
        
        raw.add_proj(projs,remove_existing = True)
        raw = raw.apply_proj()
        cov1 = mne.compute_raw_covariance(raw)
    
        # compute the variance before adding projection
        #compute the trace ratio
        tmp_trace_ratio = np.sum(np.diag(cov1.data))/np.sum(np.diag(cov0.data))
        print (tmp_trace_ratio)
        trace_ratio = np.hstack([trace_ratio, tmp_trace_ratio])
        del(raw)
   
    trace_ratio_all.append(trace_ratio.mean())
    print (i)
    

trace_ratio_all = np.array(trace_ratio_all )   
scipy.io.savemat(data_root_dir0 + "Result_Mat/SSP_proportion_var.mat",
                 dict (trace_ratio_all = trace_ratio_all))

print (1- trace_ratio_all.mean(), trace_ratio_all.std())
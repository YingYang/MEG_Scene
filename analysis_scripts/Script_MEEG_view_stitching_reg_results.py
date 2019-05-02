import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import mne
import sys
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from Stat_Utility import bootstrap_mean_array_across_subjects

subj_list = [1,2,3,4,5,6,7,8,10,11,12,13]
n_subj = len(subj_list)


matdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/MEEG_comp/stitch_data/"
test_only = True
#test_only = False
n_latents = 80

n_times = 48 # hard-coded here, I did the prediction for every 4 time points
error_ratio = np.zeros([n_subj,4,n_times])
for i in range(n_subj):
    subj = "Subj%d" % subj_list[i]
    mat_dict = scipy.io.loadmat(matdir + "%s_testonly%d_%d_latents_error_ratio.mat" % (subj, test_only, n_latents))
    tmp_error_ratio = mat_dict['error_ratio'][:,mat_dict['time_sub_ind'][0]-1]
    times = mat_dict['commonTimes'][0,mat_dict['time_sub_ind'][0]-1]
    error_ratio[i] = np.log10(tmp_error_ratio)
    
#log transform seems not that nesesary
error_ratio_no_baseline = (np.transpose(error_ratio,[2,0,1]) - np.mean(error_ratio[:,:,times<0],axis = -1)).transpose([1,2,0])   
fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/meg_eeg_sensor/"

    
    
legend_seq = ['stitch','full','eeg','meg']
plt.figure(figsize = (5,3))
# plot the mean of tmp error ratio    
plt.plot(times, np.mean(error_ratio, axis = 0).T, lw = 2)
plt.xlabel('times (ms)')
plt.ylabel('log relative error in predicting neil')
plt.legend(legend_seq)
plt.title('raw')
plt.savefig(fig_outdir + "MEG_EEG_Stitch_log_error_conv5_testonly%d.pdf" %test_only) 
    
pair_list = [[3,2],[3,0],[1,3],[1,0],[0,2],[1,2]]
n_pair = len(pair_list)
pair_names = list()
for j in range(n_pair):
    pair_names.append("%s-%s" %( legend_seq[pair_list[j][0]], legend_seq[pair_list[j][1]]))
# [baseline or not, n_pair, n_times]
    

times_in_ms = times
cluster_p_thresh = 0.05
threshold = None
ymin,ymax = -8,8

plt.figure(figsize = (12,8))
for j in range(n_pair):
    id1,id2 = pair_list[j][0], pair_list[j][1]
    tmp_diff = error_ratio[:,id1,:] - error_ratio[:,id2,:]
    tmp = bootstrap_mean_array_across_subjects(tmp_diff)
    tmp_mean = tmp['mean']
    tmp_se = tmp['se']
    ub = tmp['ub']
    lb = tmp['lb'] 

    Tobs, clusters, p_val_clusters, H0 = mne.stats.permutation_cluster_1samp_test(
                tmp_diff, threshold,tail = 0)
    print clusters, p_val_clusters           
    ax = plt.subplot(3,2,j+1)
    ax.plot(times_in_ms, Tobs)
    tmp_window = list()
    for i_c, c in enumerate(clusters):
        c = c[0]
        if p_val_clusters[i_c] <= cluster_p_thresh:
            _ = ax.axvspan(times_in_ms[c.start], times_in_ms[c.stop - 1],
                                color='r', alpha=0.2)
            _ = plt.text(times_in_ms[c.start],0.25, 
                                             ('p = %1.3f' %p_val_clusters[i_c])) 
            tmp_window.append(dict(start = c.start, stop = c.stop, p = p_val_clusters[i_c]))                              
    _ = ax.fill_between(times_in_ms, ub, lb, facecolor='b', alpha=0.4) 
    plt.xlabel('time (ms)')
    plt.ylabel('T diff')
    plt.title("%s" %(pair_names[j]))            
    plt.ylim(ymin, ymax)
plt.tight_layout()
plt.savefig(fig_outdir + "MEG_EEG_Stitch_log_error_conv5_compare_testonly%d.pdf" %test_only)


    
plt.figure()
n1,n2 = 5,5
for i in range(n_subj):
    plt.subplot(n1,n2,i+1)
    plt.plot(times, error_ratio[i].T, lw = 2)
    
#plt.legend(legend_seq)


# read AFNI/SUMA 1d.roi files, save them into freesurfer label files. 
import numpy as np
import scipy.io


# get subjects age, sex, for MEG only

subj_list = np.arange(1,19)
n_subj = len(subj_list)

age = np.zeros(n_subj)
gender = np.zeros(n_subj)
handedness = np.zeros(n_subj)

bhv_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/Experiment_mat_files/"

for i in range(n_subj):
    if subj_list[i] in [1,2,3]:
        mat_dict = scipy.io.loadmat("%sSubj%d/Subj%d_block1_run1_pre_run.mat" %(bhv_dir, subj_list[i], subj_list[i]) )
    elif subj_list[i] in [18]:
        mat_dict = scipy.io.loadmat("%sSubj%d/Original_names/Subj%d_block1_run1_MEG_pre_run.mat" %(bhv_dir, subj_list[i], subj_list[i]) )
    else:
        mat_dict = scipy.io.loadmat("%sSubj%d/Subj%d_block1_run1_MEG_pre_run.mat" %(bhv_dir, subj_list[i], subj_list[i]) )
    age[i], gender[i], handedness[i] = mat_dict['age'][0][0], mat_dict['gender'][0][0], mat_dict['handedness'][0][0]
    

print age.min(), age.max(), age.mean(), gender.sum(), handedness.sum()


EEG_subj_list = [1,2,3,4,5,6,7,8,10,11,12,13,14, 16, 18]
has_EEG = np.in1d(subj_list, EEG_subj_list)

print gender[has_EEG].sum()

#============== MEG RT and EEG RT============================================
# do box plot or mixed efect model?
# get the RT
RT_list = np.zeros([2,n_subj], dtype = np.object)
n_run_per_block = 2
for isMEG in [0,1]:
    if isMEG:
        mat_file_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/MEG_DATA/Experiment_mat_files/"
        n_runs_per_subject = [6,12,6,12,10,8,12,10,10,10,12,12,12,12,10,12,12,12]
    else:
        mat_file_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/EEG_DATA/Experiment_mat_files/"
        n_runs = 10

    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        if not isMEG and subj_list[i] not in EEG_subj_list:
            continue
        if isMEG:
            n_block = n_runs_per_subject[i]//2
        else:
            n_block = 6
        
        tmp_RT = np.zeros(0)
        is_repeated = np.zeros(0)
        for k in range(n_block):
            if isMEG:
                for l in range(n_run_per_block):
                    if subj_list[i] > 3:
                        # Subj18, order was broken: 456-123 swapped, I renamed the files with the suffix rename
                        if subj_list[i] == 18:
                            print "Subj18", subj
                            tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_MEG_post_run_rename.mat" %(k+1,l+1))
                        else:
                            tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_MEG_post_run.mat" %(k+1,l+1))        
                    else:
                        tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"/"+ subj+"_block%d_run%d_post_run.mat" %(k+1,l+1))
                    
                    if subj in ['Subj9'] and  k == 1 and l == 1:
                        tmp_n_trial = 168 
                        tmp_RT = np.hstack([tmp_RT, tmp_mat['rt'][0:tmp_n_trial,0],])
                        is_repeated = np.hstack([is_repeated, tmp_mat['this_is_repeated'][0:tmp_n_trial,0],])
                    else:
                        # use the indice 0 to 361
                        # There was a huge bug here, I add the new run on top of the old runs!! everything was wrong!!!!
                        tmp_RT = np.hstack([tmp_RT, tmp_mat['rt'][:,0]])
                        is_repeated = np.hstack([is_repeated, tmp_mat['this_is_repeated'][:,0]])
            else:
                tmp_mat = scipy.io.loadmat(mat_file_dir+subj+"_EEG/"+ "%s_block%d_EEG_post_run.mat" %(subj, k+1))
                # use the indice 0 to 361
                tmp_RT = np.hstack([tmp_RT,tmp_mat['rt'][:,0]])
                is_repeated = np.hstack([is_repeated,tmp_mat['this_is_repeated'][:,0]])
            
        tmp_RT = tmp_RT[is_repeated>0]*1000.0
        #tmp_RT[tmp_RT==0] = np.inf
        RT_list[isMEG, i] = tmp_RT[tmp_RT>0]

#
import matplotlib.pyplot as plt
plt.figure()
MEGorEEG = ['EEG','MEG']
for isMEG in range(2):
    _ = plt.subplot(2,1,isMEG+1)    
    _ = plt.boxplot(RT_list[isMEG,:])
    _ = plt.ylabel('reaction time (ms)')
    _ = plt.xlabel('participant index')
    _ = plt.title(MEGorEEG[isMEG])
    _ = plt.ylim(0,1600.0)

_= plt.tight_layout()
plt.savefig("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/Bhv/one_back_RT_box.pdf")


# mean RT
mean_RT = np.zeros([2,n_subj])
for isMEG in range(2):
    for i in range(n_subj):
        if  RT_list[isMEG,i] is not 0:
            mean_RT[isMEG,i] = np.mean(RT_list[isMEG,i])

mean_RT_across_subj = np.zeros(2)
se_RT_across_subj = np.zeros(2)
mean_RT_across_subj[0] = mean_RT[0,has_EEG].mean()
se_RT_across_subj[0] = mean_RT[0,has_EEG].std()/np.sqrt(has_EEG.sum())
mean_RT_across_subj[1] = mean_RT[1].mean()
se_RT_across_subj[1] = mean_RT[1].std()/np.sqrt(n_subj)

plt.figure(figsize = (2.5,4))
MEGorEEG = ['EEG','MEG']
index = np.array([1,2])
bar_width = 0.3
plt.bar(index, mean_RT_across_subj, bar_width, color='b', yerr=se_RT_across_subj)
_ = plt.ylim(0,1000.0)
plt.ylabel('mean reaction time (ms)')
plt.xticks(index+bar_width/2, ('EEG','MEG'))
_= plt.tight_layout()
plt.savefig("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/Bhv/one_back_RT_mean.pdf")
      
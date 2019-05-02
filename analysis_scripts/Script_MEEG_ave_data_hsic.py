import numpy as np
import scipy.io

import matplotlib
matplotlib.use('Agg')
import sys
path0 = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/"
sys.path.insert(0, path0)
path0 = "/home/ying/Dropbox/MEG_source_loc_proj/Face_Learning_Data_Ana/Utility/"
sys.path.insert(0, path0)
from Stat_Utility import bootstrap_mean_array_across_subjects



"""
When I stitched the results, offset was set to 0.02, I need to adjust an additional common offset of 0.02
"""

common_offset = 0.02

# load the results
test_name = "hsic"
feat_name = "conv5_10comp"
#feat_name = "sun"
alpha = 0.05


result_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/MEEG_comp/hsic_tests/"
fname = result_dir + "%s_with_%s_alpha%1.2f.mat" %(test_name, feat_name, alpha)

mat_dict = scipy.io.loadmat(fname)

#print mat_dict.keys()

# how many independent sampling without replacement there were for a given number of images
n_sample = mat_dict['n_sample'][0][0]
# number of permutations
n_perm = mat_dict['n_perm'][0][0]

subj_list = mat_dict['subj_list'][0]
n_subj = len(subj_list)

modality_names = mat_dict['modality_names'][0]
n_modality = len(modality_names)
for i in range(n_modality):
    modality_names[i] = modality_names[i][0]

proportion_seq = mat_dict['proportion_seq'][0]
n_proportion = len(proportion_seq)

times = (mat_dict['times'][0]-common_offset)*1000.0

subsample_index = mat_dict['subsample_index']
nLatents = mat_dict['nLatents']


# [n_modality, n_subj, n_proportion, n_sample,  n_perm, n_times]
stat_all = mat_dict['stat_all']
thresh_all = mat_dict['thresh_all']

# first MEG and EEG
# sharp ratio
sharp_ratio_ana = (stat_all/thresh_all)[:,:,:,:,0,:]

alpha1 = 0.05
# threshold based on permutations
stat_intact = stat_all[:,:,:,:,0,:]
stat_perm = stat_all[:,:,:,:,1::,:]
percentile = np.percentile(stat_perm, (1-alpha1)*100, axis = 4)
max_percentile = np.max(percentile, axis = -1)
sharp_ratio_perm_max = (stat_intact.transpose([4,0,1,2,3])/max_percentile).transpose([1,2,3,4,0])
sharp_ratio_perm_indiv = stat_intact/percentile

import matplotlib
#matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
# create colors
values = range(n_proportion+2)
jet = cm = plt.get_cmap('jet') 
cNorm  = colors.Normalize(vmin=0, vmax=values[-1])
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
colorVal_list = list()
for j in range(n_proportion):
    colorVal = scalarMap.to_rgba(values[j])
    colorVal_list.append(colorVal)



n_im_seq =(np.round(proportion_seq*362)).astype(np.int)



figoutdir= "/home/ying/Dropbox/Thesis/Dissertation/Draft/Figures/Result_figures/MEEG_comp/"

to_show_list = [(sharp_ratio_ana.mean(axis = 3)).mean(axis = 1), 
           (sharp_ratio_perm_indiv.mean(axis = 3)).mean(axis = 1),]

to_show_indiv_list = [sharp_ratio_ana.mean(axis = 3), 
                      sharp_ratio_perm_indiv.mean(axis = 3),]           
#            (sharp_ratio_perm_max.mean(axis = 3)).mean(axis = 1)]
data_name_list = ['sharp_ratio_ana', 'sharp_ratio_perm'] #, 'sharp_ratio_perm_all']

"""
# one single subject
subj_id = 3
to_show_list = [  sharp_ratio_ana.mean(axis = 3)[:,subj_id], 
             (sharp_ratio_perm_indiv.mean(axis = 3))[:,subj_id],
             (sharp_ratio_perm.mean(axis = 3))[:,subj_id]   ]
data_name_list = ['sharp_ratio_ana', 'sharp_ratio_perm', 'sharp_ratio_perm_all']
"""


figsize = (14, 3.5)
for l in range(len(to_show_list)):
    to_show = to_show_list[l]
    data_name = data_name_list[l]
    # visualize across subject
    plt.figure( figsize = figsize)
    ymin, ymax = 0.8, 1.3
    for i in range(n_modality):
        _ = plt.subplot(1, n_modality, i+1)
        for j in range(n_proportion):
            _ = plt.plot(times, to_show[i,j], lw = 2, color = colorVal_list[j])
        _ = plt.ylim(ymin, ymax)
        if i == 0:
            _ = plt.legend(n_im_seq)
        _ = plt.xlabel('time (ms)')
        _ = plt.ylabel('sharp ratio')
        _ = plt.title(modality_names[i])
        _ = plt.grid('on')
    
    _ = plt.tight_layout()
    figname = figoutdir + "%s_%s_%s.eps" %(test_name, feat_name, data_name)
    _ = plt.savefig(figname)
    _ = plt.close()
   
   

#============== plot MEG and EEG ===========
mod_ind0 = 1
prop_ind0 = 2
mod_ind1 = 0
prop_ind1 = 4
figsize = (4, 3.5)

#legend = [ "%s %s" %(modality_names[mod_ind0], n_im_seq[prop_ind0]),
#           "%s %s" %(modality_names[mod_ind1], n_im_seq[prop_ind1]), 
#           "%s %s - %s %s" %(modality_names[mod_ind0], n_im_seq[prop_ind0],
#                             modality_names[mod_ind1], n_im_seq[prop_ind1]) ]

legend = [ "%s %s" %(modality_names[mod_ind0], n_im_seq[prop_ind0]),
           "%s %s" %(modality_names[mod_ind1], n_im_seq[prop_ind1]) ]

for l in range(len(to_show_list)):
    to_show = to_show_list[l]
    data_name = data_name_list[l]
    to_show_indiv = to_show_indiv_list[l]
    # visualize across subject
    plt.figure( figsize = figsize)
    _ = plt.plot(times, to_show[mod_ind0,prop_ind0], 'r', lw = 2)
    _ = plt.plot(times, to_show[mod_ind1,prop_ind1], 'g', lw = 2)
    
    # also plot the pairwise difference
    diff = to_show_indiv[mod_ind0,prop_ind0]- to_show_indiv[mod_ind1,prop_ind1]
    mean_diff = diff.mean(axis =0)
    tmp = bootstrap_mean_array_across_subjects(diff, alpha = 0.05/len(times))
    tmp_mean = tmp['mean']
    tmp_se = tmp['se']
    #ub = tmp['ub']
    #lb = tmp['lb']
    #a0 = scipy.stats.norm.
    #ub = tmp['mean']+tmp['se']*2.0
    #lb = tmp['mean']-tmp['se']*2.0
    
    if False:
        _ = plt.plot(times, tmp_mean)
        _ = plt.fill_between(times, ub, lb, facecolor='b', alpha=0.4) 
    
    _ = plt.xlabel('time (ms)')
    _ = plt.ylabel('sharp ratio')
    _ = plt.grid('on')
    _ = plt.legend(legend)
    _ = plt.tight_layout()
    figname = figoutdir + "%s_%s_%s_compair_mod%d%d_prop%d%d.eps" \
             %(test_name, feat_name, data_name, mod_ind0, mod_ind1, prop_ind0, prop_ind1)
    _ = plt.savefig(figname)
    _ = plt.close()
   

sharp_ratio = (sharp_ratio_perm_indiv.mean(axis = 3))
time_sub_ind = np.nonzero(  np.all(np.vstack([times <=  500.0, times>= 60.0]),axis = 0) )[0]

times1 = times[time_sub_ind]
# pairwise comparison a a fixed trial number
prop_ind = 4
pair_list = [[2,0],[2,1],[3,0],[3,1],[3,2],[1,0]]

plt.figure( figsize = (20,4))
for l in range(len(pair_list)):   
    pair_ind1, pair_ind0 = pair_list[l][0],  pair_list[l][1]
    diff = sharp_ratio[pair_ind1,:,prop_ind] \
         - sharp_ratio[pair_ind0,:,prop_ind]
 
    diff = diff[:, time_sub_ind]
    plt.subplot(1, len(pair_list), l+1)

    # also plot the pairwise difference
    tmp = bootstrap_mean_array_across_subjects(diff, alpha = 0.05/len(times1))
    tmp_mean = tmp['mean']
    tmp_se = tmp['se']
    #ub = tmp['ub']
    #lb = tmp['lb']

    alpha1 = scipy.stats.norm.ppf(1-alpha/len(times1)/2.0)
    ub = tmp['mean']+tmp['se']*alpha1
    lb = tmp['mean']-tmp['se']*alpha1
    _ = plt.plot(times1, tmp_mean)
    _ = plt.fill_between(times1, ub, lb, facecolor='b', alpha=0.4) 
    _ = plt.title("%s\n-%s" %(modality_names[pair_ind1], modality_names[pair_ind0]))
    _ = plt.xticks(np.arange(100, 600, 100))
    _ = plt.xlabel('time (ms)')
    _ = plt.ylabel('sharp ratio difference')
    _ = plt.ylim(-0.2, 0.2)
    _ = plt.plot(times1, np.zeros(len(times1)), 'k')
    _ = plt.tight_layout()
    figname = figoutdir + "%s_%s_%s_compare_pairwise_propind%d.pdf" \
             %(test_name, feat_name, 'perm', prop_ind)
    _ = plt.savefig(figname)
_ = plt.close()

#===================== how many extra trials to run ========================
# ==== but what I am doing here is problematic 

# comparison betwene MEG and EEG

# to be matched, MEG
pairid0 = 1
# use EEG to match
pairid1 = 0
propid0 = 1
# find the aspect ratio that is greater than prop0

prop1 = np.zeros([n_subj, len(time_sub_ind)])
for i in range(n_subj):
    for j in range(len(time_sub_ind)):
        tmp_sharp_ratio0 = sharp_ratio[pairid0, i, propid0, time_sub_ind[j]]
        tmp_sharp_ratio1 = sharp_ratio[pairid1, i, :, time_sub_ind[j]]
        tmp = np.nonzero(tmp_sharp_ratio1 >= tmp_sharp_ratio0)[0]
        if len(tmp) == 0:
            prop1[i,j] = np.nan
        else:
            prop1[i,j] = proportion_seq[max(tmp[0], propid0)]


prop1 /= proportion_seq[propid0] 
mean_prop_ratio = np.zeros(len(time_sub_ind))
se_prop_ratio = np.zeros(len(time_sub_ind))
for j in range(len(time_sub_ind)):
    tmp_valid_ind = True - np.isnan(prop1[:,j])
    tmp_prop1 = prop1[tmp_valid_ind,j]
    mean_prop_ratio[j] = tmp_prop1.mean()
    se_prop_ratio[j] = tmp_prop1.std()/np.float(np.sqrt(len(tmp_prop1)))
 
plt.figure()
plt.errorbar(times[time_sub_ind], mean_prop_ratio, 2*se_prop_ratio)   
plt.xlabel('time (ms)')
plt.ylabel('ratio of trials')
plt.ylim(0, 4.0) 
#figname = figoutdir + "%s_%s_trial_ratio.eps" %(test_name, feat_name)
#_ = plt.savefig(figname)      


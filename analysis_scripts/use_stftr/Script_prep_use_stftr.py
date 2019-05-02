"""
Usage: python Script_use_stftr.py  Subj1
"""
import scipy.io
import numpy as np


import getpass
username = getpass.getuser()
Flag_on_cluster = True if username == "yingyan1" else False
if Flag_on_cluster:
    stc_outdir = "/data2/tarrlab/MEG_NEIL/MEG_EEG_results/MEG_stftr/"   
else:
    stc_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/"
    

# should these be the same across subjects? 
# so generate these in a different script! then always load the data
# YES, should be pre-determined and the same across subjects
n_im = 362
n_category = n_im//2
n_train_category = 90
train_category_ind = np.random.choice(range(n_category), n_train_category, replace = False)
train_im_id = np.union1d(2*train_category_ind, 2*train_category_ind+1)
test_im_id = np.setdiff1d(np.arange(0,n_im), train_im_id)

B = 500
n_test_im = len(test_im_id)
original_order = (np.arange(0,n_test_im)).astype(np.int)
bootstrap_seq = np.zeros([B+1, n_test_im], dtype = np.int)
bootstrap_seq[0] = original_order
for l in range(B):
    bootstrap_seq[l+1] = (np.random.choice(original_order, n_test_im)).astype(np.int)

mat_name = stc_outdir + "train_test_id_boostrap_seq.mat"
mat_dict = dict(train_im_id = train_im_id, test_im_id = test_im_id,
                bootstrap_seq = bootstrap_seq)
scipy.io.savemat(mat_name, mat_dict, oned_as = "row")



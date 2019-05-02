# draft use script
import scipy.io
import scipy.stats
import numpy as np
import mne
from copy import deepcopy
import pickle
import sys
import getpass
username = getpass.getuser()
Flag_on_cluster = True if username == "yingyan1" else False
if Flag_on_cluster:
    paths = ["/home/yingyan1/my_source_loc_projects/STFT_R_git_repo/evaluation/",
            "/home/yingyan1/my_source_loc_projects/STFT_R_git_repo/"]
else:
    paths = ["/home/ying/Dropbox/MEG_source_loc_proj/STFT_R_git_repo/evaluation/",
         "/home/ying/Dropbox/MEG_source_loc_proj/STFT_R_git_repo/"]
for l in range(len(paths)):
    sys.path.insert(0,paths[l])
from Evaluation_individual_G import get_solution_individual_G

#from multiprocessing import Pool
#============================ calling stftr===================================
def stftr_on_feat_L21(subj, X, fwd_path, epochs_path, datapath, sol_path,
                 train_im_id, test_im_id,
                 alpha_seq, beta_seq, gamma_seq,
                 isMEG = True, t_cov_baseline = -0.05,
                 wsize = 16, tstep = 4, 
                 maxit = 200, tol = 1E-4, Incre_Group_Numb = 1000, Maxit_J = 20, dual_tol = 0.15,
                 Flag_verbose = False, depth = 0.8,
                 label_names = None, n_active_ini = 500):
    
    """
    subj: eg "Subj1"
    X: [n_im, p], features to regress on
    fwd_path: path of the forward solution:
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                   + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
    epochs_path: path of an example epochs, e.g. 
        epochs_path = ("/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                     +"MEG_DATA/DATA/epoch_raw_data/%s/%s_%s" \
                    %(subj, subj, "run1_filter_1_50Hz_raw_ica_raw_smoothed-epo.fif.gz"))
    datapath: path of the actual sensor data
        datapath = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                  + "MEG_DATA/DATA/epoch_raw_data/%s/%s_%s.mat" \
                  %(subj, subj, fname_suffix)
    sol_path: full path to store the solutions, without the ".mat" suffix
              e.g. stc_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/"
                   sol_path = stc_outdir + "%s_STFT-R_MEG" %subj
    traim_im_id, integer indices of training images
    test_im_id, integer indices of testing images,  CAN BE None
    isMEG: if the data is from MEG
    t_cov_baseline: compute the noise covariance in -0.1 to t_cov_baseline time window
    wsize, tstep, STFT-components
    maxit, tol, Incre_Group_Numb, Maxit_J, dual_tol, optimization parameters
    Flag_verbose: if True, output optimization details
    depth = 0.8, depth parameter to reweight forward solutions
    label_names: list of full path ROI label names
        eg. label_names = ['/home/ying/dropbox_unsync/MEG_face_learning/ROI_labels/' + subj +"/" + LABEL_NAME ]
    """                 
    n_im = X.shape[0]                      
    # create evoked.list, fwd, and cov                  
    mat_data = scipy.io.loadmat(datapath)
    data = mat_data['ave_data'][:,:,:]
    data -= np.mean(data,axis = 0)
    
    n_times0 = data.shape[2]
    print n_times0
         
    # load data
    epochs = mne.read_epochs(epochs_path)
    print epochs.tmin
    tmp_tmin = epochs.tmin
    tmp_tmax = 0.01*n_times0+tmp_tmin
    # make sure the timing was correct!!
    epochs.crop(tmin = None, tmax = tmp_tmax)
    
    n_times1 = len(epochs.times)
    
    if isMEG:
        epochs1 = mne.epochs.concatenate_epochs([epochs, deepcopy(epochs)])
        epochs = epochs1[0:n_im]
        epochs._data = data[:,:,0:n_times1].copy()
        del(epochs1)
    else:
        epochs = epochs[0:n_im]
        epochs._data = data[:,:,0:n_times1].copy()
    # a temporary comvariance matrix
    # noise covariance was computed on the average across 3 to 6 repetitions, so no need to shrink it manually. 
    noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=t_cov_baseline)
    noise_cov.save(sol_path+"noise_cov-cov.fif")
    
    # after computing the noise cov, cut the baseline
    epochs.crop(tmin = t_cov_baseline, tmax = None)
    n_trials = n_im
    evoked_list = list()
    G_ind = np.zeros(n_trials, dtype = np.int)
    for r in range(n_trials):
        evoked_list.append(epochs[r].average())
        
    # load the forward solution    
    fwd = mne.read_forward_solution(fwd_path, surf_ori = True)
    fwd_list = [fwd]
    src = fwd['src']
   
    # =============== load the labels========================================
    if label_names is not None:
        labels = list()
        for i in range(len(label_names)):
            tmp_label = mne.read_label(label_names[i])
            labels.append(tmp_label)
        label_ind = list()
        for i in range(len(labels)):
            _, sel = mne.source_space.label_src_vertno_sel(labels[i],src)
            label_ind.append(sel)
    else:
        labels = None
    
    
    
    evoked_list_train = [evoked_list[i] for i in train_im_id]
    X_train = X[train_im_id,:]
    # demean X_train and X_test
    X_train = scipy.stats.zscore(X_train, axis = 0)
    G_ind_train = G_ind[train_im_id]
    
    if test_im_id is not None:
        evoked_list_test = [evoked_list[i] for i in test_im_id]
        X_test = X[test_im_id,:]
        X_test = scipy.stats.zscore(X_test, axis = 0)
        G_ind_test = G_ind[test_im_id]
#    print "X_train mean "
#    print X_train.mean(axis = 0)
#    print "X_train std"
#    print X_train.std(axis =0)
#    print "X_test mean "
#    print X_test.mean(axis = 0)
#    print "X_test std"
#    print X_test.std(axis = 0)
    
    
    delta_seq = None
    snr_tuning_seq = None
    result = get_solution_individual_G(evoked_list_train, fwd_list, G_ind_train,
                                                noise_cov, X_train, labels, 
                              alpha_seq, beta_seq, gamma_seq,
                              delta_seq = delta_seq, snr_tuning_seq = snr_tuning_seq,
                              wsize = wsize, tstep = tstep, maxit = maxit, tol = tol,
                              method = "STFT-R",
                              Incre_Group_Numb= Incre_Group_Numb, L2_option = 0,
                              ROI_weight_param=0, 
                              Maxit_J = Maxit_J, dual_tol = dual_tol, depth = depth, Flag_verbose = Flag_verbose,
                              n_active_ini = n_active_ini)
                              
    to_save = deepcopy(result)
    to_save['delta_star'] = -1
    to_save['X_train'] = X_train
    
    to_save['G_ind_train'] = G_ind_train
    
    to_save['train_im_id'] = train_im_id
    
    to_save['t_cov_baseline'] = t_cov_baseline 
    to_save['alpha_seq'] = alpha_seq
    to_save['beta_seq'] = beta_seq
    to_save['gamma_seq'] = gamma_seq                    
    
    print "%s L21 finished" %subj                          
    # save this intermediate result
    if test_im_id is not None:
        to_save['X_test'] = X_test
        to_save['G_ind_test'] = G_ind_test 
        to_save['test_im_id'] = test_im_id 
        # also save the evoked_list_test, using pickle?
        pickle_f = open(sol_path+"_test.pickle", "wb")
        pickle.dump(evoked_list_test, pickle_f, protocol = -1)
        pickle_f.close()
    
    print "saveing results"
    print sol_path
    scipy.io.savemat(sol_path + ".mat", to_save, oned_as = "row")
    

#===============================================================================    
def stftr_on_feat_L2(subj, fwd_path, sol_path, sol_outdir, bootstrap_seq, 
                 delta_seq, snr_tuning_seq = None,
                 method = "STFT-R",
                 isMEG = True, t_cov_baseline = -0.05,
                 wsize = 16, tstep = 4, 
                 maxit = 200, tol = 1E-4, 
                 Flag_verbose = False, depth = 0.8,
                 label_names = None):
    
    """
    subj: eg "Subj1"
    sol_path: full path to store the solutions
              e.g. stc_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_STC/STFTR/MEG/"
                   sol_path = stc_outdir + "%s_STFT-R_MEG_tmp.mat" %subj
    sol_outdir: outdir of the solutions 
    bootstrap_seq: list of boostrap seq, it must at least include the original sample set
    isMEG: if the data is from MEG
    t_cov_baseline: compute the noise covariance in -0.1 to t_cov_baseline time window
    wsize, tstep, STFT-components
    maxit, tol, , optimization parameters
    Flag_verbose: if True, output optimization details
    depth = 0.8, depth parameter to reweight forward solutions
    label_names: list of full path ROI label names
        eg. label_names = ['/home/ying/dropbox_unsync/MEG_face_learning/ROI_labels/' + subj +"/" + LABEL_NAME ]
    """   


    noise_cov = mne.read_cov(sol_path+"noise_cov-cov.fif") 
    evoked_list_test = pickle.load(open(sol_path+"_test.pickle", "rb"))                           
     
    mat_dict = scipy.io.loadmat(sol_path + ".mat")
    X_test = mat_dict['X_test']
    G_ind_test = mat_dict['G_ind_test'][0]
    Z_L21 = mat_dict['Z']
    coef_non_zero_mat = np.abs(mat_dict['Z']) > 0 
    active_set = mat_dict['active_set'][0]>0 
    
    # bootstrap_seq must at least include the original sample set
    n_Z = bootstrap_seq.shape[0]
    print "%d boostraps" %n_Z
    test_im_id_seq = bootstrap_seq.copy()
    n_test_im = X_test.shape[0]
    fwd = mne.read_forward_solution(fwd_path, surf_ori = True)
    fwd_list = [fwd]
    
    labels = None
    alpha_seq, beta_seq, gamma_seq = None, None, None
    times = evoked_list_test[0].times
    #========= small function for calling pool
    for l in range(n_Z):
        evoked_list_test1 = deepcopy(evoked_list_test)
        tmp_evoked_list = [evoked_list_test1[test_im_id_seq[l][l1]] for l1 in range(n_test_im) ] 
        result2 = get_solution_individual_G(tmp_evoked_list, 
                                fwd_list, G_ind_test[test_im_id_seq[l]], 
                                 noise_cov, X_test[test_im_id_seq[l]], labels, 
                              alpha_seq, beta_seq, gamma_seq = gamma_seq, 
                              delta_seq = delta_seq, snr_tuning_seq = snr_tuning_seq,
                              wsize = wsize, tstep = tstep, maxit = maxit,tol = tol,
                              method = method,
                              active_set = active_set,
                              L2_option = 2, coef_non_zero_mat =  coef_non_zero_mat,
                              Flag_verbose = Flag_verbose, depth = depth,
                              Z0_L2= Z_L21, Flag_backtrack=True, L0=1.0, eta=2)  
        result2['sample_seq'] = test_im_id_seq[l]                
        mat_name = sol_outdir + "%s_STFT-R_MEG_tmp_btstrp%d.mat" %(subj,l)
        scipy.io.savemat(mat_name, dict(Z = result2['Z'], active_set = active_set,
                         times =times, tstep = tstep, wsize = wsize,
                         delta_seq = delta_seq, delta_star = result2['delta_star'],
                         cv_MSE_L2 = result2['cv_MSE_L2'] ))
        print "Boostrap %d finished" %l 
#=====================================================================================

    

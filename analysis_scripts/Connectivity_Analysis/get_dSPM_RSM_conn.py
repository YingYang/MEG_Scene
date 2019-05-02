import numpy as np
import scipy.io

"Lagged RSM correlation"


#==============================================================================
def get_RSM(Y):
    "Y,[n_im, n_dim, n_times]"
    Y -= np.mean(Y,axis = 0)
    n_im, n_dim,n_times = Y.shape
    mask = np.ones([n_im,n_im])
    mask = np.triu(mask,1)
    RSM_ts = np.zeros([(mask>0).sum(),n_times])
    for t in range(n_times):
        tmp = np.corrcoef(Y[:,:,t])
        RSM_ts[:,t] = tmp[mask>0]
    return RSM_ts

#==============================================================================
def get_lagged_corr_RSM(RSM_tsA, RSM_tsB):
    n_times = RSM_tsA.shape[1]
    lagged_corr = np.zeros([n_times,n_times])
    for i in range(n_times):
        for j in range(n_times):
            lagged_corr[i,j] = np.corrcoef(RSM_tsA[:,i], RSM_tsB[:,j])[0,1]
    return lagged_corr

#==============================================================================
# permutation tests can be added too

if __name__ == "__main__":
    
    import mne
    import matplotlib.pyplot as plt
    import scipy.stats
    ROI_bihemi_names = [ 'pericalcarine',  #'medialorbitofrontal',
                        'PPA_c_g'] #'TOS_c_g', 'RSC_c_g']
    nROI = len(ROI_bihemi_names)                    
    labeldir = "/home/ying/dropbox_unsync/MEG_scene_neil/ROI_labels/"  
    MEGorEEG = ['EEG','MEG']
    isMEG = True
    # For now for MEG only
    stc_out_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/" \
                + "source_solution/dSPM_%s_ave_per_im/" % MEGorEEG[isMEG]
    fname_suffix = "1_110Hz_notch_ica_ave_alpha15.0"
    
    #pairs = [ [0,2],[0,3],[0,4],[2,1],[3,1],[4,1]]
    pairs = [[0,1]]
    nPair = len(pairs)

    
    times = np.arange(0.01,0.96,0.01)
    n_times = 95
    times_in_ms = times*1000.0
    mask0 = np.zeros([n_times,n_times])
    time_WOI_start = np.nonzero(times>=0.06)[0][0]
    time_WOI_end = np.nonzero(times<0.5)[0][-1]
    mask0[time_WOI_start:time_WOI_end, time_WOI_start:time_WOI_end] = 1
    mask = [ np.triu(mask0,1), np.tril(mask0,-1)]
    
    
    subj_list = np.arange(1,14)
    n_subj = len(subj_list)
    
    lagged_corr = np.zeros([n_subj, nPair, n_times, n_times])
    ul_diff = np.zeros([n_subj, nPair])
    
    for i in range(n_subj):
        subj = "Subj%d" %subj_list[i]
        labeldir1 = labeldir + "%s/" % subj
        # load and merge the labels
        labels_bihemi = list()
        for j in ROI_bihemi_names:
            tmp_label_list = list()
            for hemi in ['lh','rh']:
                print subj, j, hemi
                tmp_label_path  = labeldir1 + "%s_%s-%s.label" %(subj,j,hemi)
                tmp_label = mne.read_label(tmp_label_path)
                tmp_label_list.append(tmp_label)
            labels_bihemi.append(tmp_label_list[0]+tmp_label_list[1]) 
        fwd_path = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"\
                               + "MEG_DATA/DATA/fwd/%s/%s_ave-fwd.fif" %(subj, subj)
        fwd = mne.read_forward_solution(fwd_path, surf_ori = True) 
        src = fwd['src']
        ROI_ind = list()
        for j in range(nROI):
            tmp_label = labels_bihemi[j]
            _, tmp_src_sel = mne.source_space.label_src_vertno_sel(tmp_label, src)
            ROI_ind.append(tmp_src_sel)
        # load the source solution
        mat_dir = stc_out_dir + "%s_%s_%s_ave.mat" %(subj, MEGorEEG[isMEG],fname_suffix)
        mat_dict = scipy.io.loadmat(mat_dir)
        source_data = mat_dict['source_data']
        times = mat_dict['times'][0]
        n_times = len(times)
        del(mat_dict)
        
        ROI_data = list()
        for j in range(nROI):
            ROI_data.append(source_data[:,ROI_ind[j],:])
        del(source_data)

        ROI_RSM_ts = list()
        for j in range(nROI):
            ROI_RSM_ts.append(get_RSM(ROI_data[j]))
        
        for k in range(nPair):
            lagged_corr[i,k] = get_lagged_corr_RSM(ROI_RSM_ts[pairs[k][0]], 
                                                ROI_RSM_ts[pairs[k][1]])
        print "%s done" %subj
    
        # data for Jordan Rodu: subj = Subj10
        scipy.io.savemat('/home/ying/Dropbox/tmp/NEIL_MEG_One_Subj_V1_PPA_dSPM.mat',
                         dict(ROI_data1 = ROI_data[0],
                              ROI_data2 = ROI_data[1],
                              ROI_names = ROI_bihemi_names, subj = subj, times = times))
 


if False:        
    for i in range(n_subj):
        for k in range(nPair):
            tmp = lagged_corr[i,k]
            ul_diff[i,k] = np.mean(tmp[mask[0]>0])-np.mean(tmp[mask[1]>0])
    
    ul_diff_se = np.std(ul_diff, axis = 0)/np.sqrt(n_subj)
    T_ul_diff = np.mean(ul_diff, axis = 0)/ul_diff_se
    p_T_ul_diff = 2*(1-scipy.stats.t.cdf(np.abs(T_ul_diff), df = n_subj-1))
    
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/source_conn/"
    
    plt.figure(figsize = (14,6))
    plt.subplot(2,1,1)
    xtick = np.arange(1,nPair+1)
    width = 0.8
    plt.bar(xtick,np.mean(ul_diff, axis = 0), width = width)
    plt.errorbar(xtick+width/2, np.mean(ul_diff, axis = 0), ul_diff_se, fmt = None)
    xticknames = list()
    for k in range(nPair):
        xticknames.append("%s\n%s" % (ROI_bihemi_names[pairs[k][0]], 
                                      ROI_bihemi_names[pairs[k][1]]))
    plt.xticks(xtick+width/2,xticknames)
    plt.xlim(0.3,nPair+1)
    plt.ylabel('diff corr RSM upper-lower')
    plt.subplot(2,1,2)
    plt.bar(xtick,-np.log10(p_T_ul_diff), width = width)
    plt.xticks(xtick+width/2,xticknames)
    plt.xlim(0.3,nPair+1)
    plt.ylabel('-log10(p)')
    plt.hlines(-np.log10(0.05/nPair),0,nPair+1)
    plt.savefig(fig_outdir + "Subj_pooled_diff_u_l.pdf")
    
       
    
    vmin, vmax = -0.3, 0.3
    n1,n2 = 4,4
    for k in range(nPair):
        _=plt.figure(figsize = (14,10))
        for i in range(n_subj):
            _= plt.subplot(n1,n2,i+1)
            _ = plt.imshow(lagged_corr[i,k], aspect = "auto", interpolation = "none",
                   extent = [times_in_ms[0], times_in_ms[-1], times_in_ms[0], times_in_ms[-1]],
                    origin = "lower", vmin = vmin, vmax = vmax)
            _=plt.colorbar()
            _=plt.xlabel(ROI_bihemi_names[pairs[k][0]])
            _=plt.ylabel(ROI_bihemi_names[pairs[k][1]])
            _=plt.title("Subj%d"%subj_list[i])
        plt.tight_layout()
        plt.savefig(fig_outdir + "MEG_RSM_conn_%s_%s.pdf"
           % (ROI_bihemi_names[pairs[k][0]], ROI_bihemi_names[pairs[k][1]] ))
    
    vmin, vmax = -15,15
    T = np.zeros(lagged_corr.shape[1::])
    for k in range(nPair):
        T[k] = lagged_corr[:,k].mean(axis = 0)/lagged_corr[:,k].std(axis = 0)*np.sqrt(n_subj)
    pT = 2*(1-scipy.stats.t.cdf(np.abs(T),df = n_subj-1))
    
    reject,_ = mne.stats.fdr_correction(pT.ravel(), alpha = 0.05, method = "negcorr")
    reject = reject.reshape(pT.shape)
    _=plt.figure(figsize = (12,6))
    n0 = 2
    for k in range(nPair):
        _=plt.subplot(n0,nPair//n0,k+1)
        _ = plt.imshow(T[k]*reject[k], aspect = "auto", interpolation = "none",
               extent = [times_in_ms[0], times_in_ms[-1], times_in_ms[0], times_in_ms[-1]],
                origin = "lower", vmin = vmin, vmax = vmax)
        _=plt.colorbar()
        _=plt.xlabel(ROI_bihemi_names[pairs[k][0]])
        _=plt.ylabel(ROI_bihemi_names[pairs[k][1]])
    plt.tight_layout()
    plt.savefig(fig_outdir + "MEG_RSM_conn_T.pdf")
    
    n0 = 2
    vmin,vmax = -0.2,0.2
    _=plt.figure(figsize = (12,6))
    for k in range(nPair):
        _=plt.subplot(n0,nPair//n0,k+1)
        _ = plt.imshow(lagged_corr[:,k].mean(axis = 0), aspect = "auto", interpolation = "none",
               extent = [times_in_ms[0], times_in_ms[-1], times_in_ms[0], times_in_ms[-1]],
                origin = "lower", vmin = vmin, vmax = vmax)
        _=plt.colorbar()
        _=plt.xlabel(ROI_bihemi_names[pairs[k][0]])
        _=plt.ylabel(ROI_bihemi_names[pairs[k][1]])
    plt.tight_layout()
    plt.savefig(fig_outdir + "MEG_RSM_conn_mean.pdf")
            
        
            
        
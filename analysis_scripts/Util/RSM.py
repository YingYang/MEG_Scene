# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import mne
mne.set_log_level('WARNING')
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.spatial
#======================================================================================
def get_rsm_correlation(ave_data, X, n_perm = 50,
                     perm_seq = None, metric = "correlation", demean = True,
                     alpha = 0.05, 
                     X_rsm = None, X_rsm_perm = None):
    """
    # A time series versus a feature, not two time series with lags. 
    ave_data [n_im, n_sensor]
    X [n_im, p]
    if X_rsm is not None, then use X_rsm  = [n_im*(n_im-1)/2,]
    if X_rsm_perm is not None, it should be the permuted version corresping to the perm sequence
        X_rsm  = [n_perm, n_im*(n_im-1)/2], perm_seq is no longer needed
    """
    # checking inputs:
    if perm_seq is not None and perm_seq.shape[0]!= n_perm:
        raise ValueError("number of perms does not match perm_seq")
    if (X_rsm_perm is not None and (X_rsm_perm.shape[0]!= n_perm)):
        raise ValueError("X_rsm_perm not valid")
    
    n_im = X.shape[0]
    n_times = ave_data.shape[-1]
    if demean:
        ave_data-= np.mean(ave_data, axis = 0)
        X -= np.mean(X, axis=0)   
    if n_perm > 0:
        orig_seq = np.arange(0,n_im)
        if perm_seq is None and X_rsm_perm is None:
            perm_seq = np.zeros([n_perm, n_im])
            for k in range(n_perm):
                perm_seq[k] = np.random.permutation(orig_seq)
            perm_seq = perm_seq.astype(np.int)
        n_aug = n_perm+1  
    else:
        n_aug = 1
         
    mask = np.ones([n_im, n_im])
    mask = np.triu(mask, k=1)
    
    n_pair = n_im*(n_im-1)//2
    if X_rsm is None:
        aug_X = np.zeros([n_aug, n_im, X.shape[-1]])
        aug_X[0] = X
        if n_perm > 0:
            for k in range(n_perm):
                aug_X[k+1] = X[perm_seq[k]]
                
        RDM_tri_X_aug = np.zeros([n_aug, n_pair])
        for i in range(n_aug):
            tmp_RDM_X =scipy.spatial.distance.squareform( 
                      scipy.spatial.distance.pdist(aug_X[i],metric=metric))
            RDM_tri_X_aug[i,:] = tmp_RDM_X[mask>0]
    else:
        RDM_tri_X_aug = np.zeros([n_aug, n_pair])
        RDM_tri_X_aug[0] = X_rsm
        if n_perm > 0 and X_rsm_perm is not None:
            RDM_tri_X_aug[1::] = X_rsm_perm
    
    # 1-correlation for each time point
    RDM_sensor = np.zeros([n_times,n_im, n_im])
    for i in range(n_times):
        RDM_sensor[i] = scipy.spatial.distance.squareform( 
                  scipy.spatial.distance.pdist(ave_data[:,:,i],metric=metric))
                    
    RDM_tri_sensor = np.zeros([n_times, n_pair])
    for i in range(n_times):
        tmp = RDM_sensor[i]
        RDM_tri_sensor[i] = tmp[mask>0]
            
    corr_aug = np.zeros([n_aug,n_times])  
    for k in range(n_aug):
        for i in range(n_times):
               corr_aug[k,i] = np.corrcoef(RDM_tri_sensor[i], RDM_tri_X_aug[k])[0,1]     
    
    # TO BE ADDED
    corr_ts = corr_aug[0]
    if n_perm > 0:
        corr_ts_perm = corr_aug[1::]
        alpha_seq = [alpha/2.0*100, (1.0-alpha/2.0)*100]
        tmp = np.array(np.percentile(corr_ts_perm,alpha_seq,axis = 0))
        null_range = tmp
    else:
        corr_ts_perm = np.zeros(corr_aug[0].shape)
        null_range = np.zeros([2, n_times])
    
    result = dict(null_range = null_range, corr_ts = corr_ts, 
                  corr_ts_perm  = corr_ts_perm)
    return result
#=====================================================================

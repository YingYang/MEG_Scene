# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import time

def rbf_dot(X1, X2, deg):
    """
    https://github.com/kacperChwialkowski/wildBootstrap/blob/master/matlab/util/rbf_dot.m
    Parameters:
        patterns1, [n,d1]
        patterns2, [n,d2]
        deg, integer
    """
    n,d1 = X1.shape
    d2 = X2.shape[1]
    G = np.sum(X1**2, axis = 1)[:,np.newaxis]
    H = np.sum(X2**2, axis = 1)[:,np.newaxis]
    Q = np.tile(G,   [1,n])
    R = np.tile(H.T, [n,1])
    
    H0 = Q+R-2*X1.dot(X2.T)
    H1 = np.exp(-H0/2/deg**2)
    return H1
    

def HSIC(X,Y,deg = 1):
    """
    https://github.com/kacperChwialkowski/wildBootstrap/blob/master/matlab/wildHSIC.m
    Leila's thesis
    http://www.gatsby.ucl.ac.uk/~gretton/indepTestFiles/indep.htm   (hsicTestGamma.m), the bootstrap one computes trace(HK HL) instead
    """
    K = rbf_dot(X,X, deg = deg);
    L = rbf_dot(Y,Y, deg = deg);
    n = X.shape[0]
    H = np.eye(n)-1/n*np.ones([n,n])
    Kc = np.dot(H,K).dot(H)
    Lc = np.dot(H,L).dot(H)
    Stat = np.trace(Kc.dot(Lc))/n**2
    return Stat
    
    
def perm_test_HSIC(X,Y,perm_seq, deg = 1):
    # perm_seq [n_perm, n]
    n, d1 = X.shape
    d2 = Y.shape[1]
    n_perm = perm_seq.shape[0]
    Stat = np.zeros(n_perm+1)
    Y_aug = np.zeros([n_perm+1, n, d2])
    Y_aug[0]= Y
    for i in range(n_perm):
        Y_aug[i+1]= Y[perm_seq[i],:]
        
    for i in range(n_perm+1):
        Stat[i] = HSIC(X,Y_aug[i], deg)
    
    p_val = np.mean(Stat[0]> Stat[1::])
    if p_val == 0:
        p_val = 1.0/(n_perm+1)
    return dict(Stat = Stat[0], Stat_perm = Stat[1::], p_val = p_val)
    
  
def get_time_series_HSIC(X, Y_time, perm_seq, deg = 1):
    # Y_time [n,n_dipoles, T]
    T = Y_time.shape[-1]
    n = X.shape[0]
    
    n_perm = perm_seq.shape[0]
    n_aug = n_perm+1
    # augment X first:
    X_aug = np.zeros([n_aug, X.shape[0], X.shape[1]])
    X_aug[0] = X
    for i in range(n_perm):
        X_aug[i+1] = X[perm_seq[i],:]
        
    # compute the kernel seq of Y
    L_seq = np.zeros([T,n,n])
    for t in range(T):
        L_seq[t] = rbf_dot(Y_time[:,:,t],Y_time[:,:,t], deg = deg)
        
    K_aug = np.zeros([n_aug,n,n])
    for i in range(n_aug):
        K_aug[i] = rbf_dot(X_aug[i], X_aug[i], deg = deg)

    H = np.eye(n)-1/n*np.ones([n,n])
    Stat = np.zeros([n_aug,T])
    
    for i in range(n_aug):
        t0 = time.time()
        for t in range(T):
            K = K_aug[i]
            L = L_seq[t]
            Kc = np.dot(H,K).dot(H)
            Lc = np.dot(H,L).dot(H)
            #Stat[i,t] = np.trace(Kc.dot(Lc))/n**2
            Stat[i,t] = np.sum(Kc*Lc)/n**2
        print time.time()-t0
    
    
    tmp = np.sign(Stat[1::] - Stat[0])     
    p_time = np.mean(tmp>0,axis = 0)
    p_time[p_time==0] = 1.0/n_aug
  
    Stat_time = Stat[0]
    Stat_perm_time = Stat[1::]
    return dict(Stat_time = Stat_time, Stat_perm_time = Stat_perm_time,
                p_time = p_time)
        
    
    
    
    

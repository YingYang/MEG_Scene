# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:55:06 2014

Create epochs from all 8 blocks, threshold each trial by the meg, grad and exg change
Visulize the channel responses.

@author: ying
"""

import numpy as np
import scipy
import statsmodels.api as sm

#=======================================================================================
def ols_regression(data0, X, stats_model_flag = False):
    """
    data, [n_trials, n_dipoles/n_sensors, n_times]
    # X should not have intercept
    """
    n_im, m, n_times = data0.shape
    # demean the data
    data = data0 - np.mean(data0, axis = 0)
    F_val = np.zeros([m, n_times])
    Rsq = np.zeros([m, n_times])
    if stats_model_flag:
        # the results by two methods agree
        for i in range(m):
            if np.mod(i,1000) == 0:
                print ("%f completed" % np.float(i)/np.float(m)*100)
            for j in range(n_times):
                y = data[:,i,j]
                result = sm.OLS(y, X).fit()
                F_val[i,j] = result.fvalue
    else:
        # add the all one columne to X
        X_aug = np.zeros([n_im, X.shape[1]+1])
        X_aug[:,0] = np.ones(n_im)
        X_aug[:,1::] = X
        inv_op = (np.linalg.inv(X_aug.T.dot(X_aug))).dot(X_aug.T)
        
        for i in range(m):
            if np.mod(i,1000) == 0:
                print ("%f completed" % (np.float(i)/np.float(m)*100))
            for j in range(n_times):
                y = data[:,i,j]
                beta = np.dot(inv_op, y)
                yhat = np.dot(X_aug,beta)
                residual = y-yhat
                SSE = np.sum(residual**2)
                # total variance
                TSS = np.sum((y - np.mean(y))**2)
                # model explained variance
                SSM = TSS-SSE
                tmp_F = SSM/X.shape[1] / (SSE/(n_im - X.shape[1]))
                F_val[i,j] = tmp_F
                Rsq[i,j] = 1.0-SSE/TSS
    
    n_dim = X.shape[1]
    dfM, dfE = n_dim, n_im - n_dim-1
    log10p = np.reshape(-np.log10(1-scipy.stats.f.cdf(F_val.ravel(), dfM, dfE)), F_val.shape)    
    result = dict(F_val = F_val, log10p = log10p, Rsq = Rsq)
    return result
        


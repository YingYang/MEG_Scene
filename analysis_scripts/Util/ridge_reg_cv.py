# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 11:14:03 2015

@author: ying
"""

import numpy as np
import scipy
import sklearn.linear_model
import time

def ridge_reg_cv(X_train, X_test, Y_train, Y_test, alpha_seq, nfold = 2):
    """
    Parameters:
    # Y has pY dimensions, but the regression will be done for each dimension
    independently 
    X_train, [n_train, p]
    Y_train, [n_train, pY]
    X_test, [n_test, p]
    Y_test, [n_test, pY]
    
    # X_train and X_test should be jointly normalized before using the function
    
    Returns:
    """
    [n_train, pY] = Y_train.shape
    # to reduce running time, I can do two-fold cv
    model = sklearn.linear_model.RidgeCV(alphas=alpha_seq,
                fit_intercept=False, normalize=False, scoring=None, 
                cv= nfold, gcv_mode=None, store_cv_values=False)
    # debug
    #model = sklearn.linear_model.RidgeCV(alphas=alpha_seq,
    #            fit_intercept=True, normalize=False, scoring=None, 
    #            cv= None, gcv_mode=None, store_cv_values= True)
                
    # predict error ratio:   sum of squares of residuals/ sum of squares of Y 
    err_ratio = np.zeros(pY)
    best_alpha = np.zeros(pY)
      
    t0 = time.time()
    for i in range(pY):

        # the results should be the same, with my testing
        #model = sklearn.linear_model.RidgeCV(alphas=alpha_seq,
        #        fit_intercept=True, normalize=False, scoring=None, 
        #        cv= nfold, gcv_mode=None, store_cv_values=False)

        if np.mod(i,100) == 0:
            print i
        tmpY_train = Y_train[:,i]
        model.fit(X_train,tmpY_train)
        best_alpha[i] = model.alpha_
        #model1 = sklearn.linear_model.Ridge(alpha = best_alpha[i],
        #                                    fit_intercept = False, normalize=False)        
        # test
        tmpY_test = Y_test[:,i]
        tmpY_hat = model.fit(X_train, tmpY_train).predict(X_test)
        tmp_error = tmpY_test - tmpY_hat
        err_ratio[i] = np.sum(tmp_error**2)/np.sum(tmpY_test**2)
    
    print time.time()-t0
    return err_ratio, best_alpha
        
if __name__ == '__main__':
    # a simple simulation to test
    n = 2000
    p = 200
    pY = 100
    X = np.random.randn(n,p)
    beta = np.random.randn(p,pY)
    Y = X.dot(beta) + np.random.randn(n,pY)
    
    train_ind = np.arange(0,1500)
    test_ind = np.arange(1500, n)
    
    X_train = X[train_ind,:]
    Y_train = Y[train_ind,:]
    X_test = X[test_ind,:]
    Y_test = Y[test_ind,:]
    
    alpha_seq= 10.0**(np.arange(-4,4))
    error_ratio, best_alpha = ridge_reg_cv(X_train, X_test, Y_train, Y_test, alpha_seq)
    
    
    
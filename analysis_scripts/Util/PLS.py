import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import scipy.io
import sklearn.cross_decomposition, sklearn.cross_validation

#==============================================================================
def get_PLS_n_component_by_CV(X, Y, n_component_seq, B= 10, scale = False):
    """
    Given X,[n, p1], Y,[n, p2], use partial least sqare regression, 
    to predict Y from X, evaluate a sequence of number of components,
    by 2-fold cross validation.
    Input: 
        X [n, p1], data
        Y [n, p1], data
        n_component_seq, a sequenc of component to test
        B, repeat the two-fold cross validation for  B times
    Output:
        dict, 
        'n_comp_best', integer
        'err' error sequence for all n_components_seq
    """
    n_n_component = len(n_component_seq)
    #(1 - residual_X**2/X**2)
    score = np.zeros(n_n_component)
    for i in range(n_n_component):
        l = n_component_seq[i]
        pls = sklearn.cross_decomposition.PLSCanonical(n_components = l, 
                                                       scale = scale)
        tmp_score = 0.0        
        for k in range(B):
            tmp = sklearn.cross_validation.cross_val_score(pls,
                                     X,Y, cv=5, scoring = None)
            print l, tmp
            tmp_score += np.mean(tmp)
        tmp_score /= np.float(B)
        score[i] = tmp_score
    n_comp_best = n_component_seq[np.argmax(score)]
    return dict(n_comp_best = n_comp_best, score = score)
#============================================================================    
def get_PLS(X,Y,n_component_seq, scale = False):
    
    n_n_component = len(n_component_seq)
    score = np.zeros([2,n_n_component])  #(1 - residual_X**2/X**2)
    dict_list = list()
    for i in range(n_n_component):
        n_comp = n_component_seq[i]
        pls = sklearn.cross_decomposition.PLSCanonical(n_components = n_comp, 
                                                   scale = scale)
        pls.fit(X,Y)
        T,U = pls.x_scores_, pls.y_scores_
        P,Q = pls.x_loadings_,  pls.y_loadings_  # projection weights
        Xhat, Yhat = T.dot(P.T), U.dot(Q.T)
        resX = X-Xhat
        resY = Y-Yhat
        score[0,i] = 1.0 - np.sum(resX**2)/np.sum(X**2)
        score[1,i] = 1.0 - np.sum(resY**2)/np.sum(Y**2)
        dict_list.append(dict(n_comp = n_comp, T = T, U = U,
                              resX = resX, resY = resY))   
    return dict_list, score                            

#==============================================================================
def get_PLS_cv_error(X, Y, cv_ind, n_component_seq, B = 10, scale = False):
    """
    Input, 
    X[n,p], Y[n,q]
    cv_ind,[n], 0,1,2,...,n_fold,
    n_component_seq, sequence of number of common components
    
    Return: 3 scalars 
    dict(errY = errY,relative_errY = relative_errY, n_comp_best = n_comp_best)
    Note: X, and Y should be pre-rescaled/zscored
    """
    n_fold = np.max(cv_ind)+1
    errY = 0.0
    relative_errY = 0.0
    score = 0.0
    for l in range(n_fold):
        train_ind = np.nonzero(cv_ind != l)
        test_ind = np.nonzero(cv_ind == l)
        X_train = X[train_ind]
        Y_train = Y[train_ind]
        X_test = X[test_ind]
        Y_test = Y[test_ind]
        if len(n_component_seq) > 1:
            tmp_result = get_PLS_n_component_by_CV(X_train, Y_train, n_component_seq, B= B, scale = scale)
            n_comp_best = tmp_result['n_comp_best']
        else:
            n_comp_best = n_component_seq[0]
            #print "only one n_component given"
        pls = sklearn.cross_decomposition.PLSCanonical(n_components = n_comp_best, 
                                                   scale = scale)
        pls.fit(X,Y)
        Yhat = pls.predict(X_test)
        errY += np.sum((Yhat-Y_test)**2)
        relative_errY += np.sum((Yhat-Y_test)**2)/np.sum(Y_test**2)
        score += (1.0-pls.score(X_test,Y_test))
        # debug
        #print "score %f" % (1.0-pls.score(X_test,Y_test))
        #print "relative_errY = %f\n" % (np.sum((Yhat-Y_test)**2)/np.sum((Y_test-Y_test.mean())**2)) 
        #print ((1.0-pls.score(X_test,Y_test))- (np.sum((Yhat-Y_test)**2)/np.sum((Y_test-Y_test.mean())**2)))/ (1.0-pls.score(X_test,Y_test))
    errY = errY/np.float(n_fold)
    relative_errY = relative_errY/np.float(n_fold)
    score = score/np.float(n_fold)
    return dict(errY = errY,relative_errY = relative_errY, score = score,
                n_comp_best = n_comp_best)
                   
#==========================================================
# do SVD on resX, resY, and U intersect T
def get_PCA_of_features(X, n_comp):
    """
    Input:
        X[n,p], data
        n_comp, if integer, number of component,
                if float from 0 to 1, choose the number that explains 
                at least this much variance
    Output: 
        U[:,0:n_comp],D[0:n_comp],V[0:n_comp,:], var_per_cumsum                
    """
    # demean first
    X -= np.mean(X, axis = 0)
    U,D,V = np.linalg.svd(X, full_matrices = False)
    var_per_cumsum = np.cumsum(D**2)/np.sum(D**2)
    if n_comp < 1 and type(n_comp) == float:
        n_comp = np.nonzero(var_per_cumsum>= n_comp)[0][0]+1
    else:
        n_comp = min(np.int(n_comp), X.shape[1])
    
    U = U[:,0:n_comp]
    D = D[0:n_comp]
    V = V[0:n_comp,:]
    var_per_cumsum = var_per_cumsum[0:n_comp]
    return U, D, V, var_per_cumsum
#==========================================================
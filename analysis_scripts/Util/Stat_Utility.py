# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 14:33:18 2014

@author: ying
"""

import scipy
import numpy as np

#======================================================================
def Fisher_method(p_mat, axis = 0):
    '''
    Use Fisher's method to combined independent p-values
    Input:
        p_mat, a 2d or 1d array
        axis, the dimension to combine
    Output:
        Group_p_val
    '''
    #p_mat_vec = np.ravel(p_mat0, order = 'C')
    #p_mat_vec[p_mat_vec==0] = 1E-100
    #p_mat = np.reshape(p_mat_vec, p_mat0.shape, order = 'C')
    p_mat = -2*np.log(p_mat)
    if len(p_mat.shape) >= 2:
        chi2_stat = np.sum(p_mat, axis= axis)
        df = 2* p_mat.shape[axis]
    else:
        chi2_stat = np.sum(p_mat)
        df = 2*p_mat.shape[0]
    Group_p_val = 1-scipy.stats.chi2.cdf(chi2_stat,df) 
    return Group_p_val

#======================================================================    
    
def bootstrap_mean_array_across_subjects(data,B = 100, alpha = 0.05, method = 0):
    '''
    Data [n_subj, n_times]
    Method =0: pivotal intervals, All of Stat, Page 111, Ch8, Wasserman
    else: percentile intervals
    '''
    n_subj,n_times = data.shape
    mean_data = np.mean(data,axis = 0)
    mean_data_btstrp = np.zeros([B,n_times])
    for i in range(B):
        ind = np.random.randint(n_subj, size = n_subj)
        data_btstrp = data[ind]
        mean_data_btstrp[i] = np.mean(data_btstrp,axis =0)
        
    
    se_mean_data = np.std(mean_data_btstrp, axis = 0)
    
    # default is 95% confidence interval
    # add non_normal confidence intervals   
    # follow All of stat, chapter 8.3 Bootstrap CI,
    # but I am doing this for individual time points, so the interal is marginal
    ub = np.zeros(n_times)
    lb = np.zeros(n_times)
    
    print (alpha)
    for j in range(n_times):
        tmp = scipy.stats.mstats.mquantiles( mean_data_btstrp[:,j], prob = [alpha/2, 1-alpha/2] )
        if method == 0:
            ub[j] = 2*mean_data[j] - tmp[0]
            lb[j] = 2*mean_data[j] - tmp[1]
            # comparison of larry's lecture notes:
            pivotal =  (mean_data_btstrp[:,j]-mean_data[j])*np.sqrt(data.shape[0]) 
            tmp_t = scipy.stats.mstats.mquantiles( pivotal, prob = [alpha/2, 1-alpha/2] )
            ub[j] = mean_data[j]-tmp_t[1]/np.sqrt(data.shape[0])
            lb[j] = mean_data[j]-tmp_t[0]/np.sqrt(data.shape[0])
        
        else:
            ub[j] = tmp[1]
            lb[j] = tmp[0]
        
    if method !=0:
        print ("percentile")

    result = dict(mean = mean_data, se = se_mean_data,
                  mean_btstrp = mean_data_btstrp,
                  ub = ub, lb = lb)
    return result           
        
#========================================================================

from mne.stats.cluster_level import _find_clusters
def excursion_perm_test_1D(data, perm_data, threshold, tail = 0):
    '''
    data, [T,] 
    perm_data, [B,T]
    thershold,
    tail,
    return clusters, integral, p_val
    '''
    # merge the perm_dat and data, data is alwayse the first one
    B,T = perm_data.shape
    print (B, T)
    clusters, integral = _find_clusters(data, threshold, tail=tail)
    integral_perm = np.zeros([B,2])
    # for each row, find the clusters, and compute the summation of values within the cluster
    for i in range(B):
        _, tmp_integral = _find_clusters(perm_data[i], threshold, tail=tail)
        if len(tmp_integral):
            integral_perm[i,0], integral_perm[i,1] = tmp_integral.min(), tmp_integral.max() 
    
    if len(integral):
        p_val = np.zeros(len(integral))
        if tail == -1:
            for i in range(len(integral)):
                p_val[i] = np.mean(integral_perm[:,0] <= integral[i])
        elif tail == 1:
            for i in range(len(integral)):
                p_val[i] = np.mean(integral_perm[:,1] >= integral[i])
        else:
            for i in range(len(integral)):
                p_val[i] = np.mean(np.max(np.abs(integral_perm), axis = 1)>= np.abs(integral[i]))    
    else:
        p_val = np.ones(1)       
    return clusters, integral, p_val


#========================= a similar function, but with search multiple 1D arrays    
# Still a little weird Do not use yet

def _excurtion_perm_test_1D_multipole_array_util(data, threshold, cluster_size_threshold, tail):
    cluster_list = list()
    integral_seq = np.zeros(0)
    cluster_dim_ind = np.zeros(0)
    #print data.shape[0]
    for k in range(data.shape[0]):
        clusters, integral = _find_clusters(data[k,:], threshold, tail=tail)
        #print clusters
        if len(integral) >0:
            tmp = [c[0].stop -c[0].start >= cluster_size_threshold
                         for i_c, c in enumerate(clusters)]
            # debug
            #print tmp           
            is_cluster_large_enough = np.array(tmp)
                         
            if len(np.nonzero(is_cluster_large_enough)[0]) > 0:
                clusters = [clusters[l] for l in range(len(is_cluster_large_enough)) 
                                     if is_cluster_large_enough[l]]
                integral = integral[is_cluster_large_enough]
                #print clusters
            else:
                clusters = list()
                integral = np.zeros(0)
        if len(integral) >0: 
            #print k
            #print integral
            for l in range(len(clusters)):
                cluster_list.append(clusters[l])
            integral_seq = np.hstack([integral_seq, integral])
            cluster_dim_ind = np.hstack([cluster_dim_ind, np.ones(len(clusters))*k])
    return integral_seq, cluster_list, cluster_dim_ind




def excursion_perm_test_1D_multiple_array(data, perm_data, threshold, cluster_size_threshold, tail):
    '''
        Here, I am testing an n-dimensional time series, with the permutation tests, 
        only one-sided tests are allowed. 
        ### The idea: given certain threshold, I will get several regions distributed over the n-dimension, 
        there is some unknown dependence between the regions, therefore, we considier these regions as a family,
        and get the p-value for that the family's integral/sum of statistics, under the null(permutation)
         i.e. for each permutation, find the regions above the threshold, and larger than certain size,
            then also take the interal for all these regions too. 
        (Not correct, there will always be false +)
        ###  The idea: given certain threshold, I will get several regions distributed over the n-dimension, 
        there is some unknown dependence between the regions, therefore, we considier each region in each dimension
        as a family,  and get the p-value for that the family's integral/sum of statistics, under the null(permutation)
         i.e. for each permutation, find the regions above the threshold, and larger than certain size,
            then also take the interal for each these regions, take the maximum integral. 
        (Not correct, there will always be false +)   
    
        It is important to keep in mind that the operation on the permuted samples must be 
        exactly the same as on the orignal sample
    
    Parameters:
    data, [n, T,] 
    perm_data, [B,n,T]
    thershold, real scaler
    cluster_size_threshold, int, for each dimension, the minimal size of clusters that should be considered
    tail 1/-1,
    
    Returns:
    clusters_list,  # list of all clusters
    cluster_dim_ind, index of which dimension it was
    p_val, a single p-value

    #test code
    data = np.random.randn(2,10)
    data[0,1:4] += -2
    data[1,4:8] += -2
    perm_data = np.random.randn(40,2,10)
    threshold = -1
    tail = -1
    cluster_size_threshold = 2
    
    # how to do this?
    '''
    # merge the perm_dat and data, data is alwayse the first one
    B, n, T = perm_data.shape
    print (n, B, T)
    integral_seq, cluster_list, cluster_dim_ind = _excurtion_perm_test_1D_multipole_array_util(
                     data, threshold, cluster_size_threshold, tail)
    # permutation results               
    integral_sum_perm_abs = np.zeros(B)  
    for i in range(B):
        tmp_integral_seq, _, tmp_ind = _excurtion_perm_test_1D_multipole_array_util(
                     perm_data[i], threshold, cluster_size_threshold, tail)
        #print tmp_ind 
        if len(tmp_integral_seq) > 0:
           integral_sum_perm_abs[i] = np.max(np.abs(tmp_integral_seq))
    
    p_val = np.zeros(len(integral_seq))
    for i in range(len(integral_seq)):
        p_val[i] = np.mean(integral_sum_perm_abs >= np.abs(integral_seq[i]))        
    return cluster_list, cluster_dim_ind, p_val





#====================== may not be used =================================
import statsmodels.api as sm             
def univariate_anova_two_by_two( data, factor):
    '''
    data, [n_samples]
    factor, [n_samples,2], must be can be 0/1 or real value
    '''
    n_samples, n_factor = factor.shape
    X = np.hstack([np.ones([n_samples,1]),factor, 
                   (factor[:,0]*factor[:,1])[:,np.newaxis]])
    model = sm.OLS(data, X).fit()
    # factor 1, factor 2, interaction
    # commanted
    if False:
        import pandas as pd
        from statsmodels.formula.api import ols 
        factor = factor.astype('category')
        n_samples, n_factor = factor.shape
        tmp_dict = dict(data = data)
        for i in range(n_factor):
            tmp_dict['f%d' %i] = factor[:,i]
        
        dataframe = pd.DataFrame(tmp_dict)
       
        # Fit the model
        expression = "data ~ f0"
        for i in range(1, n_factor):
            expression += "*f%d" %i
        model = ols(expression, dataframe).fit()
    
    return model.pvalues[1:4]


        
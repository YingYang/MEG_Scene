import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
import scipy.io
import sklearn.cross_decomposition, sklearn.cross_validation

path = "/home/ying/Dropbox/Scene_MEG_EEG/analysis_scripts/Util/"
sys.path.insert(0, path)
from PLS import (get_PLS_n_component_by_CV, get_PLS, get_PCA_of_features)

model_name = "AlexNet"
#model_name = "hybridCNN"
#model_name = "placeCNN"
n_im = 362
suffix = "no_aspect"
regressor_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/Result_MAT/sensor_regression/regressor/"  
feat_nameA = "%s_conv1_%s" % (model_name, suffix)
feat_nameB = "%s_fc6_%s"   % (model_name, suffix)
#feat_nameA = "%s_fc6" %model_name
#feat_nameB = "%s_conv1"   %model_name
feat_name_seq = [feat_nameA, feat_nameB]
print feat_name_seq
n_feat_name = len(feat_name_seq) 
# load the data
n_dim = n_im
X_all = np.zeros([n_feat_name, n_im, n_dim])
var_prop = np.zeros([n_feat_name])
for j in range(n_feat_name):  
    # load the design matrix 
    feat_name = feat_name_seq[j]
    regressor_fname = regressor_dir +  "%s_PCA.mat" %(feat_name)
    tmp = scipy.io.loadmat(regressor_fname)
    X = tmp['X']*tmp['D'][0] 
    var_prop[j] = tmp['var_per_cumsum'][0,n_dim-1]
    # demean!
    X -= np.mean(X, axis = 0)
    X_all[j] = X
    
n_component_seq = (np.arange(1,60,1)).astype(np.int)
if False:
    result = get_PLS_n_component_by_CV(X_all[0], X_all[1], n_component_seq[0:6], B= 2)
    print "best_n_component= %d" % result['n_comp_best']
    print result['score']
    plt.figure()
    plt.plot(n_component_seq, result['score'])

    result_list,score = get_PLS(X_all[0], X_all[1], n_component_seq)
    plt.figure(figsize = (4.5,3.5))
    plt.plot(n_component_seq, score.T, '-*')
    plt.xlabel('d overlap')
    plt.ylabel('proportion of variance explained')
    plt.vlines(15,0,1)
    plt.tight_layout()
    fig_outdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/sensor_reg/"
    plt.savefig(fig_outdir + "%s_%s_%d_PLS_var.pdf" %(feat_nameA, feat_nameB, 15))
    
    plt.legend([feat_nameA, feat_nameB])
    score_diff = np.diff(score, axis = 1)
    score_diff = score_diff/np.max(score_diff)
    plt.plot(score_diff.T, '-^')


n_comp = 15
n_component_seq = [n_comp]
result_list,score = get_PLS(X_all[0], X_all[1], n_component_seq)
T = result_list[0]['T']
U = result_list[0]['U']
resA = result_list[0]['resX']
resB = result_list[0]['resY']
# error explained
print "ratio of resA and A %1.2f" % (np.linalg.norm(resA)**2/np.linalg.norm(X_all[0])**2)
print "ratio of resB and B %1.2f" % (np.linalg.norm(resB)**2/np.linalg.norm(X_all[1])**2)
overlap = np.hstack([T,U])
# PCA of the overlapped part
Xoverlap, junkD, junkV, junk_var_per_cumsum = get_PCA_of_features(overlap, n_comp)
var_percent = n_comp
XresA, junkDA, junkVA, var_per_cumsumA = get_PCA_of_features(resA, var_percent)
XresB, junkDB, junkVB, var_per_cumsumB = get_PCA_of_features(resB, var_percent)

mat_name = regressor_dir + "%s_%d_%s_%d_intersect_%d.mat" %(feat_nameA, XresA.shape[1], feat_nameB, XresB.shape[1], n_comp)
mat_dict = dict(Xoverlap = Xoverlap, XresA = XresA, XresB = XresB, 
                var_per_cumsumA = var_per_cumsumA, 
                var_per_cumsumB = var_per_cumsumB,
                score = score, n_component_seq = n_component_seq)
scipy.io.savemat(mat_name, mat_dict)
# this cca implementation is really confusing, I can not find the correct loading matrix
#cca = sklearn.cross_decomposition.CCA(n_components = 2,scale = False)
#X,Y = X_all[i], X_all[j]
#cca.fit(X,Y)
#proj1 = cca.x_scores_
#proj2 = cca.y_scores_
#loading1 = cca.x_loadings_ # projection weights
#loading2 = cca.y_loadings_
#weights1 = cca.x_weights_
#weights2 = cca.y_weights_
#Xhat = proj1.dot(loading1.T)
#Yhat = proj2.dot(loading2.T)
#residual1 = X-Xhat
#residual2 = Y-Yhat
#plt.figure()
#plt.subplot(2,3,1); plt.imshow(X); plt.colorbar()
#plt.subplot(2,3,2); plt.imshow(Xhat); plt.colorbar()
#plt.subplot(2,3,3); plt.imshow(residual1); plt.colorbar()
#plt.subplot(2,3,4); plt.imshow(Y); plt.colorbar()
#plt.subplot(2,3,5); plt.imshow(Yhat); plt.colorbar()
#plt.subplot(2,3,6); plt.imshow(residual2); plt.colorbar()
# but I can not reconstruct the projection using loadings and 


# Goal:to seperate two feature sets A,B to  A\B, A\intersect B,  B\A
# PLS seems to make much more sense
# X = TP^T + error
# Y = UQ^T + error
# U = \diag{D}T + error
# T, U are the scores or projections, merge them, than take the first n_comp eigen values as the common part
# Residuals are the non-overlapped parts

if False:    
    score = np.zeros([n_feat_name, n_feat_name])
    relative_error = np.zeros([n_feat_name, n_feat_name])
    n_components = 50
    m1,m2 = 10,10
    for i in range(n_feat_name):
        for j in range(n_feat_name):
            X,Y = X_all[i], X_all[j]
            pls = sklearn.cross_decomposition.PLSCanonical(n_components= n_components, scale = False)
            pls.fit(X,Y)
            proj1 = pls.x_scores_
            proj2 = pls.y_scores_
            loading1 = pls.x_loadings_ # projection weights
            loading2 = pls.y_loadings_
            weights1 = pls.x_weights_
            weights2 = pls.y_weights_
            Xhat = proj1.dot(loading1.T)
            Yhat = proj2.dot(loading2.T)
            residual1 = X-Xhat
            residual2 = Y-Yhat
            print np.linalg.norm(residual1)/np.linalg.norm(X)
            print np.linalg.norm(residual2)/np.linalg.norm(Y)
            plt.figure()
            plt.subplot(2,3,1); plt.imshow(X); plt.colorbar()
            plt.subplot(2,3,2); plt.imshow(Xhat); plt.colorbar()
            plt.subplot(2,3,3); plt.imshow(residual1); plt.colorbar()
            plt.subplot(2,3,4); plt.imshow(Y); plt.colorbar()
            plt.subplot(2,3,5); plt.imshow(Yhat); plt.colorbar()
            plt.subplot(2,3,6); plt.imshow(residual2); plt.colorbar()
            # test
            print np.linalg.norm(proj1 - X.dot(loading1))/np.linalg.norm(proj1)
            print np.linalg.norm(proj2 - Y.dot(loading2))/np.linalg.norm(proj2)
            
            corr_seq = np.zeros(n_components)
            for l in range(n_components):
                corr_seq[l] = np.corrcoef(proj1[:,l], proj2[:,l])[0,1]

            plt.figure();
            for l in range(n_components):
                plt.subplot(m1,m2,l+1)
                plt.plot(proj1[:,l], proj2[:,l], '.')
                
           
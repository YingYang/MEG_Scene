import numpy as np
import scipy.spatial
import scipy.stats
import matplotlib.pyplot as plt

n = 100
p = 4
# pA1, pB1 controls the valid dimensions in YA, YB
# pA2, pB2 controls the total dimensions in YA, YB
pA1 = 1
pA2 = 4
pB1 = 2
pB2 = 8

aA = 1.0/np.sqrt(pA2)
aB = 1.0/np.sqrt(pB2)


np.random.seed()
plt.close('all')
X = np.random.randn(n,p)
# first a few dimensions are replications of functions of X, no noise added,
# then the remaining dimensions are random noises independent of X
#YA = np.hstack([ np.tile(X, [1,pA1]), np.random.randn(n,pA2-p*pA1)])
#YB = np.hstack([ np.tile(X, [1,pB1]), np.random.randn(n,pB2-p*pB1)])
YA = np.hstack([ np.tile(X, [1,pA1]), np.random.randn(n,pA2-p*pA1)]) 
YB = np.hstack([ np.tile(X, [1,pB1]), np.random.randn(n,pB2-p*pB1)])
YA = YA + np.random.randn(n,pA2)*aA
YB = YB + np.random.randn(n,pB2)*aB

metric = "correlation"
RSM_X = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, metric = metric))
RSM_YA = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(YA, metric = metric))
RSM_YB = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(YB, metric = metric))

mask = np.ones([n,n], dtype = np.bool)
mask = np.triu(mask,1)

# correlation of RSM between Y1, Y2 and X
corr_YA = np.corrcoef(RSM_X[mask>0], RSM_YA[mask>0])[0,1]
corr_YB = np.corrcoef(RSM_X[mask>0], RSM_YB[mask>0])[0,1]
plt.figure(); plt.bar(np.arange(2), np.array([corr_YA, corr_YB]))
plt.title('correlation of RSM')

if False:
    # Kendall's tau
    tau_YA, p_YA = scipy.stats.kendalltau(RSM_X[mask>0], RSM_YA[mask>0])
    tau_YB, p_YB = scipy.stats.kendalltau(RSM_X[mask>0], RSM_YB[mask>0])
    plt.figure(); 
    plt.subplot(2,1,1)
    plt.bar(np.arange(2), np.array([tau_YA, tau_YB]))
    plt.title('Kendall tau')
    #plt.subplot(2,1,2)
    #plt.bar(np.arange(2), -np.log10(np.array([p_YA, p_YB])))
    #plt.title('-log10p of Kendall tau, \n i.i.d assumption violated, so do not interpret the p-values')

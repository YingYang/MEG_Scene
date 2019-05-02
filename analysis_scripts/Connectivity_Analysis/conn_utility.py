import numpy as np
import mne



#=================================== PLV =====================================
def get_tf_PLV(data, ind_array_tuple, demean = True, time_start = -0.1, sfreq  = 1000.0,
               fmin = 5.0, fmax = 100.0):
    """
    Get the time_frequency phase-locking value 
    Input:
        data, [n_trials, n_ROI, n_times]
        ind_array_tuple, should be (array1, array2) 
                         and  (array1[i], array2[i]) should be one pair
            e.g. if n_ROI =2, ind_array_tuple = ([0],[1])
                 if n_ROI =3, ind_array_tuple = ([0,0,1],[1,2,2])
        demean, flag, if True, take the mean across trials off. 
    Output:
        PLV [n_pairs, n_freq, n_times]
        freqs, times
        
    """
    if demean:
        data -= np.mean(data, axis = 0)
    cwt_frequencies = np.arange(fmin, fmax, 1.0)
    n_freq = len(cwt_frequencies)
    cwt_n_cycles = (np.arange(3,n_freq*0.3+3, 0.3)).astype(np.int)  
    
    PLV, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                    data, method = "plv", 
                    mode = "cwt_morlet", indices = ind_array_tuple,
                   sfreq = sfreq, fmin = fmin, fmax = fmax,
                   cwt_frequencies = cwt_frequencies,
                   cwt_n_cycles = cwt_n_cycles)
    return PLV, freqs, times+time_start  

#========================== other measurement such as coherence================
def get_stationary_spec_conn(data, ind_array_tuple, demean = True,
                             sfreq = 1000.0, method = "coh"):
    """
    Input:  data, [n_trials, n_ROI, n_times]
            ind_array_tuple, should be (array1, array2) 
                         and  (array1[i], array2[i]) should be one pair
            e.g. if n_ROI =2, ind_array_tuple = ([0],[1])
                 if n_ROI =3, ind_array_tuple = ([0,0,1],[1,2,2])
            method = "coh, wpli" coherence, weighted phase lag index, PLV also applies here.
           see document for mne.connectivity.spectral_connectivity for more choices
    Output:
          con [n_pairs, n_freq]
        freqs, times
           
    """
    if  demean:
        data -= np.mean(data, axis = 0)
    fmin = 5.0
    fmax = 50.0
    conn, freqs, times, n_epochs, n_tapers = mne.connectivity.spectral_connectivity(
                    data, method = "plv", 
                    mode = "multitaper", indices = ind_array_tuple,
                   sfreq = sfreq, fmin = fmin, fmax = fmax, mt_adaptive=True)
    return conn, freqs 

#========================== covariance of STFT components across trials =======
def get_corr_tf(data, sfreq = 1000.0, wsize = 160, tstep = 20):
    
    data -= np.mean(data,axis = 0)
    freqs = mne.time_frequency.stftfreq(wsize = wsize, sfreq = sfreq)
    n_time_steps = np.int(np.ceil(data.shape[2]/np.float(tstep)))
    n_trials, n_ROI = data.shape[0:2]
    fmax = 50.0
    n_freq = np.sum(freqs <= fmax)
    stft_data = np.zeros([n_trials, n_ROI, n_freq, n_time_steps], dtype = np.complex)
    for i in range(n_trials):
        stft_data[i] = mne.time_frequency.stft(data[i],wsize = wsize, 
                      tstep = tstep, verbose = False)[:,0:n_freq,:]
                      
    # compute the covariance
    # first dim is real and imag
    cov = np.zeros([2,n_freq, n_time_steps, n_ROI, n_ROI]) 
    corr = np.zeros([2,n_freq, n_time_steps, n_ROI, n_ROI]) 
    for i in range(n_freq):
        for j in range(n_time_steps):
            for k in range(2):
                if k == 0:
                    tmp = np.real(stft_data[:,:,i,j].T)
                else:
                    tmp = np.imag(stft_data[:,:,i,j].T)
                cov[k,i,j] = np.cov(tmp, rowvar = True) 
                #corr[k,i,j] = np.corrcoef(tmp, rowvar = True)
                corr[k,i,j] = cov[k,i,j]/\
                            np.outer( np.sqrt(np.diag(cov[k,i,j])), np.sqrt(np.diag(cov[k,i,j])) ) 
    return cov, corr, freqs[0:n_freq], tstep, wsize
#=========================== To be added, Granger Causality ==================
    

                
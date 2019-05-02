import numpy as np
import scipy.io
import sklearn.ensemble
import sklearn.neighbors

class inst_predictor:
    """
    instantaeous predictor for button press, no temporal dependence considered
    Given data, [n_trials, n_channels, n_times]
          times, [n_times,]
          RT, [n_trials,]
    Define x [n_times,] to index "probability" that a button press happen at 
    the current time point,  x[t] = exp(-||t-RT||_2^2), this class provides 
    methods to learn an instantaneous mapping from data[:,:,t-width/2:t+widht/2]
    to x[t]. 
    """
    # initialization: window width and step_size
    def __init__(self, half_width = 25, step_size = 10, gaussian_sigma = 0.05,
                 param = 15):
        self.half_width = half_width
        self.step_size = step_size
        self.gaussian_sigma = gaussian_sigma
        self.regressor = sklearn.neighbors.KNeighborsRegressor(n_neighbors = param)
                         
    
    def train(self, data, times, RT):  
        
        half_width, step_size = self.half_width, self.step_size
        gaussian_sigma = self.gaussian_sigma
                  
        
        n_trial, n_channel, n_time = data.shape
        
        # some checks
        if len(times) != n_time or n_trial != len(RT):
            raise ValueError("The data size is wrong!")
        
        # compute x
        diff_time = ((np.tile(times, [n_trial, 1])).T - RT).T
        x = np.exp(-diff_time**2/(gaussian_sigma**2)) 
        
        # seperate data and x into windows
        n_window = (n_time-(2*half_width+1))//step_size
        data_batch = np.zeros([n_trial, n_window, 
                               n_channel, 2*half_width+1])
        x_batch = np.zeros([n_trial, n_window])                      
        for i in range(n_window):
            tmp_start = i*step_size
            tmp_end = tmp_start + 2*half_width+1
            data_batch[:,i,:,:] = np.copy(data[:,:,tmp_start:tmp_end])
            x_batch[:,i] = np.mean(x[:,tmp_start:tmp_end], axis = 1)
            
        thresh = 1E-4
        x_batch[x_batch<thresh] = 0.0
        X_data = data_batch.reshape([n_trial*n_window,-1])
        Y_data = x_batch.ravel()
        #Y_data = Y_data>0
        self.regressor.fit(X_data, Y_data)

        
    def predict(self, data, times):
        half_width, step_size = self.half_width, self.step_size
        
        n_trial, n_channel, n_time = data.shape
        # some checks
        if len(times) != n_time:
            raise ValueError("The data size is wrong!")
        
        n_window = (n_time-(2*half_width+1))//step_size
        x_predict = np.zeros([n_trial, n_window])
        for i in range(n_window):
            tmp_start = i*step_size
            tmp_end = tmp_start + 2*half_width+1
            tmp_data_batch = np.copy(data[:,:,tmp_start:tmp_end])
            tmp_data_batch = np.reshape(tmp_data_batch,[n_trial, -1])
            x_predict[:,i] = self.regressor.predict(tmp_data_batch)   
        
        time_x_predict = times[half_width::step_size][0:n_window]
        return x_predict, time_x_predict   
        
    def evaluate(self, x_predict, time_x_predict, RT, threshold, tolerance_window):
        """
        Given a threshold, select the time windows above the threshold, 
        for each consecutive one, compute the middle time point, 
        if the temporal difference compared with the true RT is smaller than tolerance_window, count as correct
        otherwise, wrong. 
        threshold [0,1] 
        """
        
        
        

if __name__ == "__main__":
    isMEG = False
    for subj in ["Subj1","Subj2", "Subj3","Subj4","Subj11"]:
        #subj = "Subj2"
        meta_dir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/"
        MEG_dir = meta_dir + "MEG_DATA/DATA/epoch_raw_data/"
        EEG_dir = meta_dir + "EEG_DATA/DATA/epoch_raw_data/"
        save_name_suffix = "1_110Hz_notch_ica"
        fname_suffix = "filter_1_110Hz_notch_ica"
        sfreq = 100.0 
        epoch_dir = MEG_dir if isMEG else EEG_dir
        if isMEG:
            mat_name = epoch_dir + "%s/%s_%s_repeated_trials.mat" %(subj, subj, save_name_suffix)
        else:
            mat_name = epoch_dir + "%s_EEG/%s_%s_repeated_trials.mat" %(subj, subj, save_name_suffix)
        mat_dict = scipy.io.loadmat(mat_name)
        epoch_mat_repeat = mat_dict['epoch_mat_repeat']
        RT_repeat = mat_dict['RT_repeat'][0]
        RT_repeat[RT_repeat == 0] = 100
        times =  mat_dict['times'][0]
        
        train_data = epoch_mat_repeat[0::2,:,:]
        train_RT = RT_repeat[0::2]
        
        test_data = epoch_mat_repeat[1::2,:,:]
        test_RT = RT_repeat[1::2]
      
        predictor1 = inst_predictor(half_width = 8, step_size = 4, param = 7)
        predictor1.train(train_data, times, train_RT)
        
        x_pred_train, train_times = predictor1.predict(train_data, times)
        x_pred_test, test_times = predictor1.predict(test_data, times)
        
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt    
        figoutdir = "/home/ying/dropbox_unsync/MEG_scene_neil/MEG_EEG_DATA/figs/button_press/"
        
        for test_flag in range(2):
            plt.figure(figsize = (12,8))
            if test_flag:
                x_pred0, rt0, x_times = x_pred_test,  test_RT, test_times
                title = "test"
            else:
                x_pred0, rt0, x_times = x_pred_train, train_RT, train_times
                title = "train"
            plt.imshow(x_pred0, vmin = 0, vmax = 1, 
                       extent = [x_times[0], x_times[-1], 0, len(rt0)],
                       origin = "lower", aspect = "auto", interpolation = "none")
            plt.colorbar()           
            plt.plot( rt0, np.arange(len(rt0))+0.5, '^r')
            plt.xlim(0,2)
            plt.xlabel('time (s)')
            plt.ylabel('trial ind')
            plt.title(title)
            plt.savefig(figoutdir + "%s_isMEG%d_%s.pdf" %(subj, isMEG, title))
        
        plt.close('all')
            
        

    
    
    
    
    
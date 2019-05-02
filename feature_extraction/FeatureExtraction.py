# -*- coding: utf-8 -*-
"""
Based on 
/home/ying/Packages/caffe/caffe-master/examples/filter_visualization.ipynb

which was 
http://nbviewer.ipython.org/github/BVLC/caffe/blob/master/examples/filter_visualization.ipynb
But the online version updated, as well as the caffe package. 
My local installation of caffe was not updated, so I followed the old ipynb
"""

# change directory to the master folder/ caffe_root
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

import scipy.io
import time
import copy

# Make sure that caffe is on the python path:
caffe_root = '/home/ying/Packages/caffe/caffe-master/'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import scipy.misc

# ===================define the vidualization function========================
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    plt.imshow(data)
    
# ==================define the feature extraction function ====================
def extract_features(input_image_dir, input_image_seq, out_dir, 
                     mean_matrix, 
                     prototxt_path, 
                     model_path,
                     labels = None, togray = False,
                     crop_box = None):
    '''  
    output the results for all the following layers
        ['conv1','pool1',
         'conv2','pool2',
         'conv3',
         'conv4',
         'conv5','pool5',
         'fc6',
         'fc7',
         'prob']
    '''
    # load the model
    net = caffe.Classifier( prototxt_path, model_path)
    # doing testing instead of training                   
    net.set_phase_test()
    net.set_mode_cpu()
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    net.set_mean('data', mean_matrix)  # ImageNet mean
    net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    # For AlexNet, I need to swap to BGR
    # I need to swap BGR too for all models. 
    # the default scale is reshaping it to 227x227
    # setting the mean will cause inconsistency of shape of the matrix, weird? what to do?
    # oversample = False: take the center only
    # it should shrink the image to 227 227
    # Note the input to predict must be 0-1 float images, not int  
    n_im = len(input_image_seq)
    for j in range(n_im):
        image_path = input_image_dir + input_image_seq[j]
        im = caffe.io.load_image(image_path)
        if togray is True:
            # change the image into a gray-value image
            im = np.transpose(np.tile(np.mean(im, axis = -1),[3,1,1]),[1,2,0])
            print "togray"
        if crop_box is not None:
            im = im[crop_box[0]:crop_box[1],crop_box[2]:crop_box[3],:]
            print "cropped"
        scores = net.predict([im], oversample = False)
        if labels is not None:
            print "\n\n\n\n"
            print labels[np.argsort(-scores[0])[0:5]]
            print image_path
        #[(k, v.data.shape) for k, v in net.blobs.items()]
        result = dict(image_path = image_path)
        layers = ['conv1','pool1',
           'conv2','pool2',
           'conv3',
           'conv4',
           'conv5','pool5',
           'fc6',
           'fc7',
           'prob']
        for layername in layers:
            # since I used oversample = False, only the first crop is non zero
            result[layername] = copy.deepcopy(net.blobs[layername].data[0] ) 
        outpath = outdir +"/%s_Image%d" %(model_name, j ) + ".mat" 
        scipy.io.savemat(outpath, result)
        del(result)
    del(net)
    return 0

#=========================script ===============================================

if __name__ == '__main__':
    
    #for model_name in [ "hybridCNN","AlexNet","placeCNN","AlexNetgray","AlexNetcrop"]:    
    for model_name in [ "AlexNet"]:
    #model_name = "hybridCNN"
    #model_name = "AlexNet"
        print model_name
        if model_name == "placeCNN":
            model_dir = "/home/ying/Packages/caffe/other_models/MIT_place_CNN/%s_upgraded/" %model_name
            prototxt_path = model_dir + "places205CNN_deploy_upgraded.prototxt"
            model_path = model_dir + "places205CNN_iter_300000_upgraded.caffemodel"        
            default_mean_path =  model_dir + "places_mean.mat"
            label_path = model_dir + "categoryIndex_places205.csv"
            labels = np.loadtxt(label_path, str, delimiter='\t')
            # this was [256,256,3], should be transformed. 
            mean_matrix = np.transpose(scipy.io.loadmat(default_mean_path)['image_mean'],[2,0,1])
            # The top 5 labels seem to work quite accurate, I tested 3 or 4.
            # so the channel swapping and transpose of the mean is correct.
            togray = False
            crop_box = None
    
        if model_name == "hybridCNN":
            # not verified
            model_dir = "/home/ying/Packages/caffe/other_models/MIT_place_CNN/%s_upgraded/" %model_name
            prototxt_path = model_dir +"hybridCNN_deploy_upgraded.prototxt"
            model_path = model_dir +"hybridCNN_iter_700000_upgraded.caffemodel"        
            default_mean_path = model_dir + "hybridCNN_mean.binaryproto"
            # http://sites.duke.edu/rachelmemo/2015/04/30/convert-binaryproto-to-npy/
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open( default_mean_path , 'rb' ).read()
            blob.ParseFromString(data)
            mean_matrix = np.array( caffe.io.blobproto_to_array(blob) )
            mean_matrix = mean_matrix[0]        
            label_path = model_dir+"categoryIndex_hybridCNN.csv"
            labels = np.loadtxt(label_path, str, delimiter='\t')
            togray = False
            crop_box = None
            
            #mean_matrix = np.transpose(scipy.io.loadmat(default_mean_path)['image_mean'],[2,0,1])            
        if model_name == "AlexNet":
            #http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
            prototxt_path = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
            model_path =  caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
            label_path = caffe_root + 'data/ilsvrc12/synset_words.txt'       
            #if using AlexNet, the default mean path is npy file, and works 
            labels = np.loadtxt(label_path, str, delimiter='\t')
            default_mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
            mean_matrix = np.load(default_mean_path)
            #also in AlexNet,
            togray = False
            crop_box = None
            
        
        if model_name == "AlexNetgray":
            #http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
            prototxt_path = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
            model_path =  caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
            label_path = caffe_root + 'data/ilsvrc12/synset_words.txt'       
            #if using AlexNet, the default mean path is npy file, and works 
            labels = np.loadtxt(label_path, str, delimiter='\t')
            default_mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
            mean_matrix = np.load(default_mean_path)
            togray = True
            crop_box = None
            
        if model_name == "AlexNetcrop":
            prototxt_path = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
            model_path =  caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
            label_path = caffe_root + 'data/ilsvrc12/synset_words.txt'       
            #if using AlexNet, the default mean path is npy file, and works 
            labels = np.loadtxt(label_path, str, delimiter='\t')
            default_mean_path = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
            mean_matrix = np.load(default_mean_path)
            
            togray = False
            mat_data = scipy.io.loadmat('/home/ying/Dropbox/Scene_MEG_EEG/Features/selected_image_second_round_data.mat');
            aspect_ratio = mat_data['aspect_ratio'] [:,0]
            n_im = len(aspect_ratio)
            # width, hight
            im_size = np.zeros([n_im, 2])  
            max_side = 500.0
            full_side = 600.0
            for i in range(n_im):
                if aspect_ratio[i] >= 1:
                    im_size[i,0] = max_side
                    im_size[i,1] = max_side/aspect_ratio[i]
                else:
                    im_size[i,0] = max_side*aspect_ratio[i]
                    im_size[i,1] = max_side
            half_width = np.min(im_size)//2       
            crop_box = (np.array([(full_side)/2-half_width,\
            (full_side)/2+half_width,\
            (full_side)/2-half_width,\
            (full_side)/2+half_width])).astype(np.int)

            
    if False:   
        outdir = "/home/ying/Dropbox/Scene_MEG_EEG/Features/%s_Features/%s_Image_Features" % (model_name, model_name)
        image_path_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Image_List.txt"
        input_image_seq = np.loadtxt(image_path_fname, delimiter = " ", dtype = np.str_)
        input_image_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Images/"
        extract_features(input_image_dir, input_image_seq, outdir, 
                         mean_matrix, 
                         prototxt_path, 
                         model_path,
                         labels = labels, 
                         togray = togray,
                         crop_box = crop_box)
                         
        
    
        # not used images
        outdir = "/home/ying/Dropbox/Scene_MEG_EEG/Features/%s_Features/%s_Extra_Image_Features/" % (model_name, model_name)
        image_path_fname = "/home/ying/Dropbox/Scene_MEG_EEG/Extra_Images/Extra_Images_list.txt"
        input_image_seq = np.loadtxt(image_path_fname, delimiter = " ", dtype = np.str_)
        input_image_dir = "/home/ying/Dropbox/Scene_MEG_EEG/Extra_Images/Extra_Images_boxed/"
        extract_features(input_image_dir, input_image_seq, outdir, 
                         mean_matrix, 
                         prototxt_path, 
                         model_path,
                         labels = labels, 
                         togray = togray,
                         crop_box = crop_box)
    
    #================ save layer one weights
    net = caffe.Classifier( prototxt_path, model_path)
    # doing testing instead of training                   
    net.set_phase_test()
    net.set_mode_cpu()
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    net.set_mean('data', mean_matrix)  # ImageNet mean
    net.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    net.set_channel_swap('data', (2,1,0))
    # size of filters [11,11], stride of 4, input 224x224
    filters = net.params['conv1'][0].data
    matname = "/home/ying/Dropbox/Scene_MEG_EEG/Features/%s_%s_filters.mat" %(model_name, "conv1")
    scipy.io.savemat(matname, dict(filters = filters))
    
    net.blobs['data'].data[0].shape





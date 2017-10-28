import caffe

import numpy as np
import scipy.io as spio
from PIL import Image, ImageDraw

import random


class HSSegDataLayer(caffe.Layer):
    """
    Load input image and label image and reshape
    the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - data_dir: path to data
        - size: size of sample
        - split: train / val / test
        - mean: tuple of mean values to subtract
        - seed: seed for randomization (default: None / current time)

        example

        params = dict(data_dir="/path/to/data",
            mean=(104.00698793, 116.66876762, 122.67891434),
            split="val")
        """
        # config
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.label_dir = params['label_dir']
        self.batch_size = params['batch_size']
        self.size = params['size']
        # self.split = params['split']
        # self.mean = np.array(params['mean'])
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.image = spio.loadmat(self.data_dir)['paviaU']
        self.labelimage = spio.loadmat(self.label_dir)['paviaU_gt']
	
	
        self.idx = 0
        self.idy = 0

	# load image + label image pair
        self.data = self.load_image(self.idx, self.idy, self.size)
        self.label = self.load_label(self.idx, self.idy, self.size)


    def reshape(self, bottom, top):
        #print self.label
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(self.batch_size, *self.data.shape)
        top[1].reshape(self.batch_size, *self.label.shape)

    def forward(self, bottom, top):
        for itt in range(self.batch_size):
            # load image + label image pair
            self.data = self.load_image(self.idx, self.idy, self.size)
            self.label = self.load_label(self.idx, self.idy, self.size)

            # assign output
            top[0].data[itt,...] = self.data
            top[1].data[itt,...] = self.label

            # pick next input
            self.idx = random.randint(0, self.image.shape[0] - 1 - self.size)
            self.idy = random.randint(0, self.image.shape[1] - 1 - self.size)
            #self.idx = 4
            #self.idx = 90

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, idx, idy, size):
        """
        Load input mat and preprocess for Caffe:
        - cast to float
        - subtract mean
        - transpose to channel x height x width order
        """
        in_ = np.array(self.image, dtype=np.uint8)
        # Use random section
        
        in_ = in_[idx:idx + size, idy:idy + size, :]
        in_ = in_[:, :, ::-1]
        # in_2 -= self.mean



        # Draw rect on testing location
        # padded = np.pad(in_, 2, 'constant', constant_values=0)
        # show_image = Image.fromarray(padded[:,:,50])
        # draw = ImageDraw.Draw(show_image)
        # draw.rectangle([idx+size-1, idy+size-1, idx, idy], fill='BLACK')
        # show_image.save('paviau.bmp')

        in_ = in_.transpose((2, 0, 1))
        # print idx, idx + size - 1, idy, idy + size - 1, in_.shape
        # print in_2.shape
        #in_3 = np.concatenate([in_, np.zeros((2, self.size, self.size))])
        # print in_3.shape
        return in_

    def load_label(self, idx, idy, size):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        label = np.array(self.labelimage, dtype=np.uint8)
        label2 = label[np.newaxis, ...]
        label2 = label2[:, idx:idx + size, idy:idy + size]

        # Draw rect on testing location for debug
        # padded = np.pad(label, 2, 'constant', constant_values=0)
        # show_image = Image.fromarray(padded)
        # draw = ImageDraw.Draw(show_image)
        # draw.rectangle([idx+size-1, idy+size-1, idx, idy], fill='WHITE')
        # show_image.save('paviaul.gif')

        return label2


class AttentionLayer(caffe.Layer):

    def setup(self, bottom, top):
                # one top: attentioned
        if len(top) != 1:
            raise Exception("Need to define one top: attentioned.")
        # one bottom: data
        if len(bottom) != 1:
            raise Exception("Need to define one bottom: data.")
            params = eval(self.param_str)
            self.W = params['W']
            self.lr = params['lr']

        def reshape(self, bottom, top):
            if top[0].shape != bottom[0].shape:
                print "Shapes are wrong"

        def forward(self, bottom, top):
            top[0].data[...] = bottom[0]

        def backward(self, top, propagate_down, bottom):
            bottom[0].diff = np.multiply(self.W, top[0].diff)
            self.W = self.W - (top[0].diff * lr)

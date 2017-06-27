import caffe

import numpy as np
import scipy.io as spio
from PIL import Image

import random

class HSSegDataLayer(caffe.Layer):
    """
    Load input image and label image and 
    reshape the net to preserve dimensions.

    Use this to feed data to a fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Setup data layer according to parameters:

        - data_dir: path to data
        - size: number of cuts to make
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
        self.split = params['split']
        self.mean = np.array(params['mean'])
        self.seed = params.get('seed', None)

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        split_f  = '{}/ImageSets/Segmentation/{}.txt'.format(self.voc_dir,
                self.split)
        self.indices = open(split_f, 'r').read().splitlines()
        self.idx = 0

    def reshape(self, top):
        # load image + label image pair
        self.data = self.load_image(self.indices[self.idx])
        self.label = self.load_label(self.indices[self.idx])
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)


    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.indices)-1)
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0


    def backward(self, top, propagate_down, bottom):
        pass


    def load_image(self, idx):
        """
        Load input mat and preprocess for Caffe:
        - cast to float
        - subtract mean
        - transpose to channel x height x width order
        """
        im = spio.loadmat(self.data_dir)
        in_ = np.array(im['paviau'], dtype=np.uint8)
        in_ = in_[:,:,::-1]
        in_ -= self.mean
        in_ = in_.transpose((2,0,1))
        return in_


    def load_label(self, idx):
        """
        Load label image as 1 x height x width integer array of label indices.
        The leading singleton dimension is required by the loss.
        """
        im = spio.loadmat(self.label_dir)
        label = np.array(im['paviau_gt'], dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label

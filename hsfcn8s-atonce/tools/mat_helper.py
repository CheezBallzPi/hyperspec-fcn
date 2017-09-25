import os
import copy
import glob
import numpy as np
import scipy.io as spio

from PIL import Image


class hypermat:

    def __init__(self, data_path, label_path):
        # data_path is /path/to/pavia
        self.immat = spio.loadmat(data_path)['paviaU']
        print self.immat.shape
        self.lbmat = spio.loadmat(label_path)['paviaU_gt']

        # lbmat = lbmat[np.newaxis, ...]
        self.classes = ['Asphalt','Meadows','Gravel','Trees',
                      'Painted metal sheets','Bare Soil',
                      'Bitumen','Self-Blocking Bricks','Shadows']
        # self.classes = classes
        
    
    def load_band(self, band):
        return self.immat[:,:,band]
        
    def load_image(self):
        return self.immat
        
    def load_labels(self):
        return self.lbmat
    

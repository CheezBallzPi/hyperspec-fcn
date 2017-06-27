import os
import copy
import glob
import numpy as np
import scipy.io as spio

from PIL import Image

class pavia:
    def __init__(self, data_path):
    	# data_path is /path/to/pavia
		immat = spio.loadmat('{}/PaviaU.mat'.format(data_path))
		lbmat = spio.loadmat('{}/PaviaU_gt.mat'.format(data_path))
    	lbmat = lbmat[np.newaxis, ...]
    	self.classes = ['Asphalt','Meadows','Gravel','Trees',
    					'Painted metal sheets','Bare Soil',
    					'Bitumen','Self-Blocking Bricks','Shadows']
    
    def load_band(self, band):
    	return immat[:,:,band]
    	
    def load_labels(self):
    	return lbmat

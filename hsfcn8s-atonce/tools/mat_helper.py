import os
import copy
import glob
import numpy as np
import scipy.io as spio

from PIL import Image

class hypermat:
    def __init__(self, data_path, label_path, classes):
    	# data_path is /path/to/pavia
		immat = spio.loadmat(data_path)
		lbmat = spio.loadmat(label_path)
    	lbmat = lbmat[np.newaxis, ...]
    	# self.classes = ['Asphalt','Meadows','Gravel','Trees',
    	# 				'Painted metal sheets','Bare Soil',
    	# 				'Bitumen','Self-Blocking Bricks','Shadows']
    	self.classes = classes
    
    def load_band(self, band):
    	return immat[:,:,band]
    	
    def load_image(self):
    	return immat
    	
    def load_labels(self):
    	return lbmat

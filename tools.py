import numpy as np
from keras.utils import np_utils
import objgraph
import sys

np.random.seed(1337)

def zeropad(X, offset, y=None):
    zerosX = np.zeros((X.shape[0] + offset*2, X.shape[1] + offset*2, X.shape[2]))
    zerosX[offset : X.shape[0] + offset, offset : X.shape[1] + offset, : ] = X
    if y is not None:
        zerosY = np.zeros((y.shape[0] + offset*2, y.shape[1] + offset*2))
        zerosY[offset : y.shape[0] + offset, offset : y.shape[1] + offset] = y
    return (zerosX, zerosY) if y is not None else zerosX

def make_sample(img, x, y, window):
    ''' Given an (X, Y, Z), returns a (window, window, Z) '''
    return img[x : x + window, y : y + window, :]

def make_sample_2d(img, x, y, window):
    ''' Given an (X, Y, Z), returns a (window, window, Z) 2d '''
    return img[x : x + window, y : y + window]

def virt_consts():
    ''' Creates the constants that are used in virt(). '''
    return (np.random.random(), 0.04*np.random.normal(0, 1))

def virt(smp1, smp2):
    ''' Makes virtual sample by changing radiation level. '''
    # new = a(old) + bn
    a1 = np.random.random()
    a2 = np.random.random()
    return (a1 * smp1 + a2 * smp2) / (a1 + a2) + 0.04 * np.random.normal(0, 1)

def reshape(smp):
    ''' Reshapes into (103,27,27,1) '''
    X = np.transpose(smp, (2,0,1))
    X = X[..., np.newaxis]
    return X

def reshape2(smp):
    ''' Reshapes into (1, 103,27,27,1) '''
    X = np.transpose(smp, (2,0,1))
    X = X[np.newaxis, ..., np.newaxis]
    return X

def get_samples(X, y, num, window):
    ''' Gets num samples. '''
    count = 0
    X_samples = np.ndarray((num, window, window, X.shape[2]))
    y_samples = np.ndarray((num, window, window))
    while(count != num):
        (xpos, ypos) = (int(np.random.randint(0, X.shape[0] - window - 1)),
                        int(np.random.randint(0, X.shape[1] - window - 1)))
        # Instead of making actual images, return the positions instead.
        X_samples[count] = make_sample(X, xpos, ypos, window) #+ np.random.random()
        y_samples[count] = make_sample_2d(y, xpos, ypos, window)
        count += 1
    return (X_samples, y_samples)

def make_batch(X, y, num, window):
    ''' Converts img into a batch of all the data points to train on.
    Samples are 15 of each class.
    Also makes validation sets. '''
    
    #X_train = np.ndarray((classes, 103, window, window, 1))
    #y_train = np.ndarray((classes))
    #X_val = np.ndarray((classes, 103, window, window, 1))
    #y_val = np.ndarray((classes))

    print(X.shape)
    print(window)
    X_train, y_train = get_samples(X, y, num[0], window)
    X_val, y_val = get_samples(X, y, num[1], window)

    return X_train, y_train, X_val, y_val
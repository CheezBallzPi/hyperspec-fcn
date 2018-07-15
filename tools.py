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
    return img[x : x + window - 1, y : y + window - 1, :]

def make_sample_2d(img, x, y, window):
    ''' Given an (X, Y, Z), returns a (window, window, Z) 2d '''
    return img[x : x + window - 1, y : y + window - 1]

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

def get_samples(X, y, num, classes, window): 
    ''' Gets num samples from each class. '''
    halfwindow = int(window / 2)
    count = np.zeros(classes)
    samples = np.ndarray((classes,num,window,window,103))
    while(count.min() != num):
        (xpos, ypos) = (np.random.randint(0, X.shape[0] - window),
                        np.random.randint(0, X.shape[1] - window))
        label = int(y[xpos + halfwindow, ypos + halfwindow])
        if(label == 0):
            continue
        if(count[label - 1] != num):
            samples[label - 1, int(count[label - 1])] = make_sample(X, xpos, ypos, window)
            count[label - 1] += 1
            #print('ylabel:', y[xpos, ypos])
        #print(count)
    #print(samples)

    return samples

def get_samples_const(X, num, window):
    ''' Gets num samples. '''
    count = 0
    samples = np.ndarray((num, 2))
    while(count != num):
        (xpos, ypos) = (int(np.random.randint(0, X.shape[0] - window - 1)),
                        int(np.random.randint(0, X.shape[1] - window - 1)))
        # Instead of making actual images, return the positions instead.
        samples[count, :] = (xpos, ypos)
        count += 1
    return samples

def make_batch(X, y, num, window):
    ''' Converts img into a batch of all the data points to train on.
    Samples are 15 of each class.
    Also makes validation sets. '''
    
    #X_train = np.ndarray((classes, 103, window, win    dow, 1))
    #y_train = np.ndarray((classes))
    #X_val = np.ndarray((classes, 103, window, window, 1))
    #y_val = np.ndarray((classes))

    print(X.shape)
    print(window)
    X_train = get_samples_const(X, num[0], window)
    X_val = get_samples_const(X, num[1], window)

    X_cat = np.concatenate((X_train, X_val))
    print(X_cat[0])
    y_cat = [make_sample_2d(y, x[0], x[1], window) for x in X_cat.astype(int)]

    print(X_cat.dtype)
    print(X_cat, y_cat[0])

    return X_cat, y_cat, (num[1] / (num[1] + num[0]))
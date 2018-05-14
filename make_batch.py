import numpy as np
from keras.utils import np_utils
import objgraph
import sys

np.random.seed(1337)

def make_sample(img, x, y, window):
    ''' Returns a (27,27,103) '''
    hw = int(window / 2)
    sample = img[x - hw : x + hw + 1, y - hw : y + hw + 1, :]
    return sample

def virt_consts():
    ''' Creates the constants that are used in virt(). '''
    return (np.random.random(), 0.01*np.random.normal(0, 1))


def virt(smp1, smp2, a, b):
    ''' Makes virtual sample by changing radiation level. '''
    # new = a(old) + bn
    return np.dot(a,smp1) + np.dot((1 - a), smp2) + b

def reshape(smp):
    ''' Reshapes into (103,27,27,1) '''
    X = np.transpose(smp, (2,0,1))
    X = X[..., np.newaxis]
    return X

def get_samples(X, y, num, classes, window):
    ''' Gets num samples from each class. '''
    halfwindow = int(window / 2)
    count = np.zeros(classes)
    samples = np.ndarray((classes,num,window,window,103))
    print(count)
    while(count.min() != num):
        (xpos, ypos) = (np.random.randint(halfwindow, X.shape[0] - halfwindow),
                        np.random.randint(halfwindow, X.shape[1] - halfwindow))
        label = int(y[xpos, ypos])
        if(label == 0):
            continue
        if(count[label - 1] != num):
            samples[label - 1, int(count[label - 1])] = make_sample(X, xpos, ypos, window)
            # Instead of making actual images, return the positions instead.
            #samples[label - 1, int(count[label - 1])] = (xpos, ypos)
            count[label - 1] += 1
            #print('ylabel:', y[xpos, ypos])
        #print(count)
    #print(samples)

    return samples

def make_batch(X, y, num, classes, window):
    ''' Converts img into a batch of all the data points to train on.
    Samples are 15 of each class.
    Also makes validation sets. '''
    
    #X_train = np.ndarray((classes, 103, window, win    dow, 1))
    #y_train = np.ndarray((classes))
    #X_val = np.ndarray((classes, 103, window, window, 1))
    #y_val = np.ndarray((classes))
    
    X_train = []
    y_train = []
    X_val = []
    y_val = []

    print(X.shape)
    print(window)
    samples_t = get_samples(X, y, 15, classes, window)
    samples_v = get_samples(X, y, 15, classes, window)
    print(samples_t.shape, samples_v.shape)

    # TODO: save random seeds instead of random images

    for n in range(classes):
        print("Class: " + str(n))
        for _ in range(num[0][n]):
            X_train.append(reshape(virt(samples_t[n, np.random.randint(0, 14)], 
                                        samples_t[n, np.random.randint(0, 14)],
                                        virt_consts()[0], virt_consts()[1])))
            #X_train.append(samples_t[n, np.random.randint(0, 14)], samples_t[n, np.random.randint(0, 14)], window, virt_consts())
            y_train.append(n)
            
    #print(X_train.shape, y_train.shape)
    print("val")
    for n in range(classes):
        print("Class: " + str(n))
        for _ in range(num[1][n]):
            X_val.append(reshape(virt(samples_v[n, np.random.randint(0, 14)], 
                                      samples_v[n, np.random.randint(0, 14)],
                                      virt_consts()[0], virt_consts()[1])))
            #X_val.append(samples_t[n, np.random.randint(0, 14)], samples_t[n, np.random.randint(0, 14)], window, virt_consts())
            y_val.append(n)
            
    y_train = np_utils.to_categorical(y_train, 9)
    y_val = np_utils.to_categorical(y_val, 9)
    print(len(X_train), len(y_train))
    #print(y_train, y_val)

    return (np.concatenate((X_train,X_val)), np.concatenate((y_train,y_val)), (sum(num[1])) / ((sum(num[1])) + sum(num[0])))
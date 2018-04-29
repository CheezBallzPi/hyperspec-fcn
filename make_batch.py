import numpy as np
from keras.utils import np_utils

np.random.seed(1337)

def make_sample(img, x, y, window):
    # Returns a (27,27,103)
    hw = int(window / 2);
    sample = img[x - hw : x + hw + 1, y - hw : y + hw + 1, :]
    return sample

def virt(smp1, smp2):
    # Makes virtual sample by changing radiation level
    # new = a(old) + bn
    (a, b) = (np.random.random(), 0.01*np.random.normal(0, 1))
    return np.dot(a,smp1) + np.dot((1 - a), smp2) + b

def reshape(smp):
    # Reshapes into (1,103,27,27,1)
    X = np.transpose(smp, (2,0,1))
    X = X[np.newaxis, ..., np.newaxis]
    return X

def get_samples(X, y, num, classes, window):
    # Gets num samples from each class.
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
            count[label - 1] += 1
            #print('ylabel:', y[xpos, ypos])
        #print(count)
    #print(samples)

    return samples


def make_batch(X, y, numt, numv, classes, window):
    # Converts img into a batch of all the data points to train on
    # Samples are 15 of each class.
    # Also makes validation sets
    # Numv is number of images PER CLASS
    halfwindow = int(window / 2)
    X_train = np.ndarray((numt, 103, window, window, 1))
    y_train = np.ndarray((numt))
    X_val = np.ndarray((numv * classes, 103, window, window, 1))
    y_val = np.ndarray((numv * classes))

    print(X.shape)
    print(window)
    samples_t = get_samples(X, y, 15, classes, window)
    samples_v = get_samples(X, y, numv, classes, window)
    print(samples_t.shape, samples_v.shape)

    for n in range(numt):
        rand_class = np.random.randint(0, classes)
        new = reshape(virt(samples_t[rand_class, np.random.randint(0, 14)], 
                           samples_t[rand_class, np.random.randint(0, 14)]))
        X_train[n] = new
        y_train[n] = rand_class
    print(X_train.shape, y_train.shape)
    count = 0
    for n in range(classes):
        for m in range(numv):
            X_val[count] = reshape(samples_v[n, m])
            y_val[count] = n
            count+=1
    print(X_val.shape, y_val.shape)
    y_train = np_utils.to_categorical(y_train, 9)
    y_val = np_utils.to_categorical(y_val, 9)
    print(X_train.shape, y_train.shape)
    print(y_train, y_val)

    return (np.concatenate((X_train,X_val)), np.concatenate((y_train,y_val)), (numv * classes) / ((numv * classes) + numt))


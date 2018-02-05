import numpy as np
from keras.utils import np_utils

np.random.seed(1337)

def make_sample(img, x, y):
    # Returns a (27,27,103)
    sample = img[x-13:x+14,y-13:y+14,:]
    return sample

def mix_virt(smp1, smp2):
    # Makes virtual sample using
    # Mu
    a = make_sample(img, x1, y1)
    b = make_sample(img, x2, y2)
    new = (a + b)

def rad_virt(smp):
    # Makes virtual sample by changing radiation level
    # new = a(old) + bn
    (a, b) = (np.random.random(), np.random.normal(0, 400))
    return (a * smp) + b

def reshape(smp):
    # Reshapes into (1,103,27,27,1)
    X = np.transpose(smp, (2,0,1))
    X = X[np.newaxis, ..., np.newaxis]
    return X

def make_batch(X, y, num):
    # Converts img into a batch of all the data points to train on

    X_batch = np.ndarray((num,103,27,27,1))
    y_batch = np.ndarray((num))
    print(X.shape)
    for n in range(num):
        while True:
            (xpos, ypos) = (np.random.randint(13, X.shape[0]-14), np.random.randint(13, X.shape[1]-14))
            if y[xpos, ypos] != 0:
                break;
        new = reshape(rad_virt((make_sample(X, xpos, ypos))))
        X_batch[n] = new
        y_batch[n] = y[xpos, ypos]
        if y[xpos, ypos] == 0:
            print(y[xpos, ypos])
        if n % 1000 == 0:
            print(n)
    print(X_batch.shape, y_batch.shape)
    y_batch = np_utils.to_categorical(y_batch, 10)
    print(X_batch.shape, y_batch.shape)
    return (X_batch, y_batch)


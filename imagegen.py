import os, sys
import scipy.io as sp
import numpy as np
import tools as t

X = sp.loadmat("data/pavia/PaviaU.mat")['paviaU']
y = sp.loadmat("data/pavia/PaviaU_gt.mat")['paviaU_gt']

num = (3000, 3000)
window = 30

X_mean = np.mean(X, axis=(0,1))
X_new = X - X_mean
X = X_new / (np.amax(X_new, axis=(0,1)) - np.amin(X_new, axis=(0,1)))

X_train, y_train, X_val, y_val = t.make_batch(X, y, num, window)

sp.savemat("./data/gen/Pavia_train.mat", dict(X=X_train, y=y_train))
sp.savemat("./data/gen/Pavia_val.mat", dict(X=X_val, y=y_val))
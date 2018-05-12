import net, make_batch as mb
import keras, numpy as np
import scipy.io as sp

np.random.seed(1337)

model = net.net()

X = sp.loadmat("data/pavia/PaviaU.mat")['paviaU']
#X = np.ones((4,4,4))
y = sp.loadmat("data/pavia/PaviaU_gt.mat")['paviaU_gt']

#classnum = [[548,  540,   392,  542,  256,  532,  375, 514,  231],
#            [5472, 13750, 1331, 2573, 1122, 4572, 981, 3363, 776]]

classnum = [[54,  54,   39,  54,  25,  53,  37, 51,  23],
            [547, 1375, 133, 257, 112, 457, 98, 336, 77]]

#mean subtraction and normalization
X_mean = np.mean(X, axis=(0,1))
X = X - X_mean
X = X / (np.amax(X, axis=(0,1)) - np.amin(X, axis=(0,1)))

# 2 options here, either 0 pad by 13 around the data
# or just never test with the outside border,

# 0 Pad option
offset = 13
zerosX = np.zeros((X.shape[0] + offset*2, X.shape[1] + offset*2, X.shape[2]))
zerosX[offset : X.shape[0] + offset, offset : X.shape[1] + offset, : ] = X
X = zerosX
zerosY = np.zeros((y.shape[0] + offset*2, y.shape[1] + offset*2))
zerosY[offset : y.shape[0] + offset, offset : y.shape[1] + offset] = y
y = zerosY
print(X.shape, y.shape)

#mb.get_samples(X, y, 15, 9, 27)

# Make batches
(X_full, y_full, split) = mb.make_batch(X, y, classnum, 9, 27)
print(X_full.shape, y_full.shape, split)

# Train!
logger = keras.callbacks.CSVLogger("logs/train.log")
cp = keras.callbacks.ModelCheckpoint("./cp/weights.{epoch:02d}.hdf5")
model.fit(X_full, y_full, batch_size=32, nb_epoch=10, validation_split=split, callbacks=[logger,cp])
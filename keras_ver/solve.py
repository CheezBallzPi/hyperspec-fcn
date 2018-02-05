import net, make_batch as mb
import keras, numpy as np
import scipy.io as sp

np.random.seed(1337)

model = net.net()

X = sp.loadmat("../data/pavia/PaviaU.mat")['paviaU']
#X = np.ones((4,4,4))
y = sp.loadmat("../data/pavia/PaviaU_gt.mat")['paviaU_gt']

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

# Make batch
(X_train, y_train) = mb.make_batch(X, y, 10000)

# Train!
logger = keras.callbacks.CSVLogger("train.log")
cp = keras.callbacks.ModelCheckpoint("./cp/weights.{epoch:02d}.hdf5")
model.fit(X_train, y_train, batch_size=32, nb_epoch=10, callbacks=[logger,cp])
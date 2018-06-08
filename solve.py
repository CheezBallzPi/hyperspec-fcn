import net, predict, tools
import keras, numpy as np
import scipy.io as sp

np.random.seed(1337)

model = net.net11()

X = sp.loadmat("data/pavia/PaviaU.mat")['paviaU']
#X = np.ones((4,4,4))
y = sp.loadmat("data/pavia/PaviaU_gt.mat")['paviaU_gt']

classnum = [[548,  540,   392,  542,  256,  532,  375, 514,  231],
            [5472, 13750, 1331, 2573, 1122, 4572, 981, 3363, 776]]

#classnum = [[54,  54,   39,  54,  25,  53,  37, 51,  23],
#            [547, 1375, 133, 257, 112, 457, 98, 336, 77]]

#mean subtraction and normalization
X_mean = np.mean(X, axis=(0,1))
X_new = X - X_mean
X_new = X_new / (np.amax(X_new, axis=(0,1)) - np.amin(X_new, axis=(0,1)))

# 2 options here, either 0 pad by 13 around the data
# or just never test with the outside border,

# 0 Pad option
# X, y = tools.zeropad(X, y, 13)
# New offset due to smaller patch
print(X_new.shape, y.shape)
X_new, y_new = tools.zeropad(X_new, 5, y=y)
print(X_new.shape, y_new.shape)

#tools.get_samples(X, y, 15, 9, 27)

# Make batches
(X_full, y_full, split) = tools.make_batch(X_new, y_new, classnum, 9, 11)
print(X_full.shape, y_full.shape, split)

# Train!    
logger = keras.callbacks.CSVLogger("logs/train.log")
cp = keras.callbacks.ModelCheckpoint("./cp/weights.{epoch:02d}.hdf5")
model.fit(X_full, y_full, batch_size=32, epochs=15, validation_split=split, callbacks=[logger,cp])

print("Predicting")
p = predict.pred(X, "cp/weights.15.hdf5")
print(p[3])
i_p = predict.toImg(p, y.shape)
i_gt = predict.toImg(y, y.shape, palette=i_p.getpalette())
i_p.save("pred.png")
i_gt.save("ground.png")
predict.compare(i_p, i_gt).save("comp.png")
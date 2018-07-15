import net, predict, tools
import keras, numpy as np
import scipy.io as sp

np.random.seed(4123)

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

#tools.get_samples(X, y, 15, 9, 27)

# Make batches 
(X_full, y_full, split) = tools.make_batch(X_new, y, (5000, 50000), 11)
print(X_full.shape)
# Train!    
model = net.netconv(X_new)

logger = keras.callbacks.CSVLogger("logs/train.log")
cp = keras.callbacks.ModelCheckpoint("./cp/weights.{epoch:02d}.hdf5")
model.fit(X_full, y_full, batch_size=32, epochs=25, validation_split=split, callbacks=[logger,cp])

print("Predicting")
p = predict.pred(X, model)
i_p = predict.toImg(p, y.shape)
i_gt = predict.toImg(y, y.shape, palette=i_p.getpalette()[1:])
print(list(i_p.getdata())[100:130])
print(list(i_gt.getdata())[100:130])
print(i_p.getpalette())
print(i_gt.getpalette())
i_p.save("pred.png")
i_gt.save("ground.png")
predict.compare(i_p, i_gt).save("comp.png")
import net, predict, tools as t
import keras, numpy as np
import scipy.io as sp

np.random.seed(4123)
X = sp.loadmat("data/pavia/PaviaU.mat")["paviaU"]
y = sp.loadmat("data/pavia/PaviaU_gt.mat")["paviaU_gt"]

X_train = sp.loadmat("data/gen/Pavia_train.mat")['X']
y_train = sp.loadmat("data/gen/Pavia_train.mat")['y']
X_val = sp.loadmat("data/gen/Pavia_val.mat")['X']
y_val = sp.loadmat("data/gen/Pavia_val.mat")['y']
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
classnum = [[548,  540,   392,  542,  256,  532,  375, 514,  231],
            [5472, 13750, 1331, 2573, 1122, 4572, 981, 3363, 776]]

# Train!    
model = net.netconv()

logger = keras.callbacks.CSVLogger("logs/train.log")
cp = keras.callbacks.ModelCheckpoint("./cp/weights.{epoch:02d}.hdf5")
X_full = np.concatenate((X_train, X_val))
y_full = np.concatenate((y_train, y_val))
print(X_full.shape, y_full.shape)
model.fit(X_full, 
          y_full, 
          batch_size=32, 
          epochs=25, 
          validation_split=.5, 
          callbacks=[logger,cp])

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
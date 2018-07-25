import keras.models as md
import tools as t
import numpy as np
from numpy import random as rd
from PIL import Image


def pred(img, model):
    # Zero pad data
    X = t.zeropad(img, 5)
    data = np.ndarray((img.shape[0] * img.shape[1],
                       103, 11, 11, 1), dtype=np.int32)
    count = 0
    for w in range(img.shape[0]):
        for h in range(img.shape[1]):
            smp = t.reshape(t.make_sample(X, w+5, h+5, 11))
            data[count] = smp
            count += 1
    print("Start")
    return model.predict_classes(data, verbose=1)
    # return [np.argmax(m.predict(t.reshape2(z))) + 1 for z in data]


def toImg(pred, shape, palette=None):
    arr = np.asarray(pred)
    arr = arr.reshape(shape)
    print(arr, arr.shape)
    #img=Image.fromarray(np.subtract(255, np.divide(255, arr).astype(int)))
    img = Image.fromarray(arr.astype(int))
    img = img.convert(mode='P')
    if not palette:
        print("Generating New Palette")
        palette = [rd.randint(0, 256) for _ in range(10 * 3)]
    palette[0:2] = [0, 0, 0]
    print(palette)
    img.putpalette(palette)
    return img


def compare(imga, imgb):
    a = np.asarray(imga)
    b = np.asarray(imgb)
    print(a.shape, b.shape)
    if a.shape != b.shape:
        return None
    c = np.zeros_like(a)
    for x in range(a.shape[0]):
        for y in range(a.shape[1]):
            c[x, y] = 0 if a[x, y] == 0 or b[x, y] == 0 else (
                1 if a[x, y] == b[x, y] else 2)
    fin = Image.fromarray(c, 'P')
    fin.putpalette([0, 0, 0,
                    0, 254, 0,
                    254, 0, 0, ])
    return fin

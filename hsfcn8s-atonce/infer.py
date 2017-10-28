import numpy as np
from PIL import Image
import copy
import tools.mat_helper as mh
from matplotlib import pyplot
import caffe

# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
print mh.hypermat('../data/pavia/PaviaU.mat','../data/pavia/PaviaU_gt.mat').load_image().shape
in_ = mh.hypermat('../data/pavia/PaviaU.mat','../data/pavia/PaviaU_gt.mat').load_image()
print in_.shape
#in_ = np.concatenate([in_, np.zeros((610, 340, 2))], axis=2)
#in_ = in_[4:4 + 30, 90:90 + 30, :]
in_ = in_[:,:,::-1]
in_ = in_.transpose((2,0,1))


# load net
net = caffe.Net('deploy.prototxt', 'snapshot/train_iter_20000.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
out = net.blobs['upscore'].data[0].argmax(axis=0).squeeze()
# np.savetxt("out.txt",net.blobs['score'].data[0,:,:,:])
out_8 = np.empty_like(out, dtype=np.uint8)
np.copyto(out_8, out, casting='unsafe')
out_8 = out_8
img = Image.fromarray(out_8)
img.save("infer_out.png")
#pyplot.imshow(out_8)
#pyplot.imshow(out)
print out, out_8
#print net.params['conv1_1'][0].data[...]
#print net.params['conv2_1'][0].data[...]
#print net.params['conv3_1'][0].data[...]
#print net.params['conv4_1'][0].data[...]
#print net.params['conv5_1'][0].data[...]
#print net.params['fc6'][0].data[...]
print net.params['fc7'][0].data[...]
#print net.params['upscore2'].data[...]

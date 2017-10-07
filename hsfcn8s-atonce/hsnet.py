import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom, nout, ks, stride=1, pad=1, group=1):
	if group==1:
		conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	else:
		conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, group=group, pad=pad,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def fcn(split):
    n = caffe.NetSpec()
    pydata_params = dict(seed=1337, size=30, split=split)
    # --- TODO: FIX THIS --- #
    pydata_params['data_dir'] = '../data/pavia/PaviaU.mat'
    pydata_params['label_dir'] = '../data/pavia/PaviaU_gt.mat'
    if split == 'train':
        pydata_params['sbdd_dir'] = '../data/sbdd/dataset'
        pylayer = 'HSSegDataLayer'
    else:
        pydata_params['voc_dir'] = '../data/pascal/VOC2011'
        pylayer = 'HSSegDataLayer'
    n.data, n.label = L.Python(module='tools.layers', layer=pylayer,
            ntop=2, param_str=str(pydata_params))
	# ---------------------- #

    # the base net
    n.conv1, n.relu1 = conv_relu(n.data, 32, 4)
    n.pool1 = max_pool(n.relu1)

    n.conv2, n.relu2 = conv_relu(n.pool1, 64, 5)
    n.pool2 = max_pool(n.relu2)

    n.conv2, n.relu2 = conv_relu(n.score, 128, 4)

    # fully conv
    #n.fc6, n.relu6 = conv_relu(n.pool2, 4096, ks=7, pad=0)
    #n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    #n.fc7, n.relu7 = conv_relu(n.drop6, 4096, ks=1, pad=0)
    #n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    #n.score = crop(n.upscore8, n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()

def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train')))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn('val')))

if __name__ == '__main__':
    make_net()

import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop


def conv_relu(bottom, nout, ks, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride, num_output=nout, pad=pad,
                             param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
                             weight_filler=dict(type='xavier'))
    return conv, L.ReLU(conv, in_place=True)


def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def fcn(split):
    n = caffe.NetSpec()
    pydata_params = dict(seed=1337, size=27, split=split)

    pydata_params['data_dir'] = '../data/pavia/PaviaU.mat'
    pydata_params['label_dir'] = '../data/pavia/PaviaU_gt.mat'
    pydata_params['batch_size'] = 10
    n.data, n.label = L.Python(module='tools.layers', layer='HSSegDataLayer',
                               ntop=2, param_str=str(pydata_params))


    # the base net
    n.conv1, n.relu1 = conv_relu(n.data, 32, 4)
    n.pool1 = max_pool(n.relu1)

    n.conv2, n.relu2 = conv_relu(n.pool1, 64, 5)
    n.pool2 = max_pool(n.relu2)
    n.drop2 = L.Dropout(n.pool2, dropout_ratio=0.5, in_place=True)

    n.conv3, n.relu3 = conv_relu(n.drop2, 128, 4)
    n.drop3 = L.Dropout(n.relu3, dropout_ratio=0.5, in_place=True)

    # fully conv
    n.fc7, n.relu7 = conv_relu(n.drop3, 729, ks=3, pad=0)
    #n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)

    n.score_fr = L.Convolution(n.relu7, num_output=9, kernel_size=1, pad=0, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], weight_filler=dict(type='xavier'))
    
    n.upscore = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=9, kernel_size=25, stride=1,
            bias_term=False),
        param=[dict(lr_mult=0)])

    n.loss = L.SoftmaxWithLoss(n.upscore, n.label,
                               loss_param=dict(normalize=False, ignore_label=255))

    return n.to_proto()


def make_net():
    with open('train.prototxt', 'w') as f:
        f.write(str(fcn('train')))

    with open('val.prototxt', 'w') as f:
        f.write(str(fcn('val')))


if __name__ == '__main__':
    make_net()

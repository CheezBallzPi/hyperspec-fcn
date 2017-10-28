import caffe
import tools.surgery, tools.hsscore, hsnet

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass
hsnet.make_net()
weights = '../data/ilsvrc-nets/VGG_ILSVRC_16_layers.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

neta = caffe.Net('train.prototxt', caffe.TEST)
solver = caffe.SGDSolver('solver.prototxt')

tools.surgery.transplant(solver.net,neta)
del neta

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
tools.surgery.interp(solver.net, interp_layers)

for _ in range(75):
    solver.step(4000)
    # scoring
    tools.hsscore.seg_tests(solver, False, 1000, layer='upscore')
    # do a visualization

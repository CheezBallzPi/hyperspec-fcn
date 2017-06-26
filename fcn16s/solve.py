import caffe
import tools.surgery, tools.score

import numpy as np
import os
import sys

try:
    import setproctitle
    setproctitle.setproctitle(os.path.basename(os.getcwd()))
except:
    pass

weights = '../fcn32s/snapshot/train_iter_24000.caffemodel'

# init
caffe.set_device(0)
caffe.set_mode_gpu()

solver = caffe.SGDSolver('solver.prototxt')
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
tools.surgery.interp(solver.net, interp_layers)

# scoring
val = np.loadtxt('../data/VOC/seg11valid.txt', dtype=str)

for _ in range(25):
    solver.step(4000)
    tools.score.seg_tests(solver, False, val, layer='score')

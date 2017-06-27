# HyperSpectral FCN
This is an implementation of FCN and Attention layers to recognize HyperSpectral images.

## Prerequisites
- Python 2.7
  - Numpy
  - Caffe
  - Scipy
- An Nvidia GPU (Not required but recommended)
- A Training and Validation dataset
- VGG model trained weights

## Setup
Anaconda is recommended for Python and its prerequisites. This implementation uses Python 2.7 so be sure to install the right one.
You can install Anaconda from [here](https://www.continuum.io/downloads).

The models are currently made to train from the VOC 2012 dataset and SBD dataset. Put these datasets in `data/VOC` and `data/sbdd` respectively.

You can get the trained weights for the VGG model on the Caffe Model Zoo or [right here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md). Put the `.caffemodel` file in `data/ilsvrc-nets`.

## Training
To train this model, run `solve.py` in `fcn32s`, then in `fcn16s`, and finally in `fcn8s`. Alternatively, run `solve.py` in `fcn8s-atonce`.

You can change what model the 16s and 8s models run off of in `solve.py` by replacing:

```
weights = '../fcn16s/snapshot/train_iter_8000.caffemodel'
```
with the model you want to train with:

```
weights = '../path/to/your/model.caffemodel'
```

## Todo
- *Make module to import HyperSpectral images*
- Add the Attention Layer
- Optional CPU/GPU
- Visualization

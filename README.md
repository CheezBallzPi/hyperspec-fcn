# HyperSpectral FCN
This is an implementation of FCN and Attention layers to recognize HyperSpectral images.

## Prerequisites
- Python 2.7
  - Caffe 
- An Nvidia GPU (Not required but recommended)
- A Hyperspectral image (currently only Pavia University Dataset works)

## Setup
Anaconda is recommended for Python and its prerequisites. This implementation uses Python 2.7 so be sure to install the right one.
You can install Anaconda from [here](https://www.continuum.io/downloads).

The models are currently made to train from the Pavia University dataset. Put this image in `data/pavia`.

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
- Add the Attention Layer
- Visualization

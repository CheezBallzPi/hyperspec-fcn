# HyperSpectral FCN
This is an implementation of FCN and Attention layers to recognize HyperSpectral images.

## Prerequisites
- Python 3
  - Keras 
- An Nvidia GPU (Not required but recommended)
- A Hyperspectral image (currently only Pavia University Dataset works)

## Setup
Anaconda is recommended for Python and its prerequisites. This implementation uses Python 3 so be sure to install the right one.
You can install Anaconda from [here](https://www.continuum.io/downloads).

The models are currently made to train from the Pavia University dataset. Put this image in `data/pavia`.

## Training
To train this model, run `solve.py`.

## Todo
- Add the Attention Layer
- **Visualization**

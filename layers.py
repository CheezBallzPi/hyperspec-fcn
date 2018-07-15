from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tools

class GenSample(Layer):
    def __init__(self, image, window, **kwargs):
        self.image = image
        self.window = window
        super(GenSample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GenSample, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        print(x.eval(session=K.get_session()))
        return self.image[int(x[:, 0]) : int(x[:, 0]) + self.window, int(x[:, 1]) : int(x[:, 1]) + self.window]

    def compute_output_shape(self, input_shape):
        return (self.window, self.window, self.image.shape[2])
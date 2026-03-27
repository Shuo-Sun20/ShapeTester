import keras
import numpy as np

def call_func(inputs, axis=-1, mean=None, variance=None, invert=False):
    norm_layer = keras.layers.Normalization(axis=axis, mean=mean, variance=variance, invert=invert)
    return norm_layer(inputs)

example_input = np.random.randn(3, 5).astype('float32')
example_output = call_func(example_input, axis=-1)
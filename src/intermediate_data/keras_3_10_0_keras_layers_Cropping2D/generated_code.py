import keras
import numpy as np

def call_func(inputs, cropping=((0, 0), (0, 0)), data_format=None):
    layer = keras.layers.Cropping2D(cropping=cropping, data_format=data_format)
    return layer(inputs)

example_input = np.random.randn(2, 28, 28, 3).astype(np.float32)
example_output = call_func(example_input, cropping=((2, 2), (4, 4)))
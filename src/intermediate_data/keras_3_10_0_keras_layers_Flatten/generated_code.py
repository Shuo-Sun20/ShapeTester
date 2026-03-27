import keras
import numpy as np

def call_func(inputs, data_format=None):
    flatten_layer = keras.layers.Flatten(data_format=data_format)
    return flatten_layer(inputs)

x = np.random.random((2, 10, 64)).astype('float32')
example_output = call_func(x)